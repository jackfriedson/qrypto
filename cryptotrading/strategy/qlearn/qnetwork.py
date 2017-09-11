import logging
import random
import time
from collections import deque, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cryptotrading.backtest import Backtest
from cryptotrading.data.datasets import QLearnDataset
from cryptotrading.data.indicators import BasicIndicator
from cryptotrading.strategy.qlearn.qestimator import QEstimator, ModelParametersCopier


log = logging.getLogger(__name__)


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])


class QNetworkStrategy(object):

    def __init__(self, exchange, base_currency: str, quote_currency: str,
                 unit: float, ohlc_interval: int = 5, train_start: str = '6/1/2017',
                 train_end: str = '7/1/2017', sleep_duration: int = 5, **kwargs) -> None:
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.unit = unit
        self.ohlc_interval = ohlc_interval
        self.exchange = exchange
        self.exchange_train = Backtest(exchange, base_currency, quote_currency, start=train_start,
                                       end=train_end, interval=ohlc_interval)
        self.exchange = None

        indicators = []
        indicators.append(BasicIndicator('macd', {'fastperiod': 10, 'slowperiod': 26, 'signalperiod': 9}))
        self.data = QLearnDataset(indicators, **kwargs)
        self.data.update(self.exchange_train.all())
        self.data.normalize()

    def update(self):
        new_data = self.exchange.get_ohlc(self.base_currency, self.quote_currency, interval=self.ohlc_interval)
        self.data.update(new_data)

    def train(self,
              gamma: float = 0.95,
              n_epochs: int = 10,
              n_hidden_units: int = 5,
              random_seed: int = 12345,
              epsilon_start: float = 1.,
              epsilon_end: float = 0.,
              epsilon_decay: float = 2.,
              validation_percent: float = 0.2,
              replay_memory_start_size: int = 1000,
              replay_memory_max_size: int = 10000,
              replay_memory_batch_size: int = 8,
              update_target_every: int = 1000):

        total_steps = len(self.exchange_train.date_range)
        train_steps = int((1. - validation_percent) * total_steps)
        validation_steps = int(validation_percent * total_steps) - 1

        tf.reset_default_graph()
        if random_seed:
            tf.set_random_seed(random_seed)

        tf.Variable(0, name='global_step', trainable=False)

        q_estimator = QEstimator('q_estimator', self.data.n_state_factors, n_hidden_units, self.data.n_actions)
        target_estimator = QEstimator('target_q', self.data.n_state_factors, n_hidden_units, self.data.n_actions)
        estimator_copy = ModelParametersCopier(q_estimator, target_estimator)

        epsilon = tf.train.polynomial_decay(epsilon_start, tf.contrib.framework.get_global_step(),
                                            train_steps * (n_epochs - 1), end_learning_rate=epsilon_end,
                                            power=epsilon_decay)
        policy = self.make_epsilon_greedy_policy(q_estimator, epsilon, self.data.n_actions)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            replay_memory = deque(maxlen=replay_memory_max_size)
            self.data.start_training()

            print('Initializing replay memory...')
            for i in range(replay_memory_start_size):
                state = self.data.state()
                action = policy(sess, state)
                reward = self.data.step(action)
                next_state = self.data.state()
                replay_memory.append(Transition(state, action, reward, next_state))

            for epoch in range(n_epochs):
                print('\nEpoch {}'.format(epoch))
                print('  Global step: {}'.format(sess.run(tf.contrib.framework.get_global_step())))
                print('  Randomness: {}%'.format(int(100 * sess.run(epsilon))))

                skipped_states = self.data.start_training()
                losses = []

                # Train the model
                for i in range(train_steps - skipped_states):

                    global_step = sess.run(tf.contrib.framework.get_global_step())
                    if (global_step // replay_memory_batch_size) % update_target_every == 0:
                        estimator_copy.make(sess)

                    state = self.data.state()
                    action = policy(sess, state)
                    reward = self.data.step(action)
                    next_state = self.data.state()

                    replay_memory.append(Transition(state, action, reward, next_state))

                    samples = random.sample(replay_memory, replay_memory_batch_size)
                    states_batch, action_batch, reward_batch, next_states_batch = map(np.array, zip(*samples))

                    q_values_next = target_estimator.predict(sess, next_states_batch)
                    targets_batch = reward_batch + gamma * np.amax(q_values_next, axis=1)

                    loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)
                    losses.append(loss)

                avg_loss = sum(losses) / len(losses)
                print('  Average loss: {}'.format(avg_loss))

                # Evaluate the model
                rewards = []
                returns = []
                confidences = []
                start_price = self.data.last

                for i in range(validation_steps):
                    state = self.data.state()
                    q_values = q_estimator.predict(sess, np.expand_dims(state, 0))[0]
                    best_action = np.argmax(q_values)
                    reward, cum_return = self.data.step_val(best_action)

                    rewards.append(reward)
                    returns.append(cum_return)
                    confidences.append((np.abs(q_values[0] - q_values[1])) / (np.abs(q_values[0]) + np.abs(q_values[1])))

                rewards = list(filter(lambda x: x != 0, rewards))
                returns = list(filter(lambda x: x != 0, returns))
                avg_reward = sum(rewards) / (len(rewards) or 1.)
                avg_confidence = sum(confidences) / len(confidences)

                print('  Average reward: {:.4f}%'.format(100 * avg_reward))
                print('  Average confidence: {:.2f}%'.format(100 * avg_confidence))

                position_value = start_price
                for return_val in returns:
                    position_value *= 1 + return_val
                market_return = (self.data.last / start_price) - 1.
                algorithm_return = (position_value / start_price) - 1.

                print('  Market return: {:.2f}%'.format(100 * market_return))
                print('  Outperformance: {:+.2f}%'.format(100 * (algorithm_return - market_return)))

                if epoch % 2 == 0 or epoch == n_epochs - 1:
                    self.data.plot(show=False, save_file='{}__{}.png'.format(int(time.time()), epoch))

    def make_epsilon_greedy_policy(self, estimator, epsilon, n_actions):
        def policy_fn(sess, observation):
            epsilon_val = sess.run(epsilon)
            action_probs = np.ones(n_actions, dtype=float) * epsilon_val / n_actions
            q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
            best_action = np.argmax(q_values)
            action_probs[best_action] += (1.0 - epsilon_val)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            return action
        return policy_fn

    def run(self):
        pass
