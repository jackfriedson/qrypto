import logging
import time
from collections import deque, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cryptotrading.backtest import Backtest
from cryptotrading.data.datasets import QLearnDataset
from cryptotrading.data.indicators import BasicIndicator
from cryptotrading.strategy.qlearn.qestimator import QEstimator


log = logging.getLogger(__name__)


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])


def make_epsilon_greedy_policy(estimator, epsilon, n_actions):
    def policy_fn(sess, observation):
        epsilon_val = sess.run(epsilon)
        A = np.ones(n_actions, dtype=float) * epsilon_val / n_actions
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon_val)
        return A
    return policy_fn


def make_greedy_policy(estimator, n_actions):
    def policy_fn(sess, observation):
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)

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
              learn_rate: float = 0.2,
              gamma: float = 0.25,
              n_epochs: int = 20,
              n_hidden_units: int = 3,
              random_seed: int = 12345,
              epsilon_start: float = 1.,
              epsilon_end: float = 0.,
              epsilon_decay: float = 2.,
              validation_percent: float = 0.2,
              replay_memory_max_size: int = 500000,
              replay_memory_start_size: int = 50000):

        total_steps = len(self.exchange_train.date_range)
        train_steps = int((1. - validation_percent) * total_steps)
        validation_steps = int(validation_percent * total_steps) - 1

        replay_memory = deque(maxlen=replay_memory_max_size)

        tf.reset_default_graph()
        if random_seed:
            tf.set_random_seed(random_seed)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        q_estimator = QEstimator('q_estimator', self.data.n_state_factors, n_hidden_units, self.data.n_actions)
        target_estimator = QEstimator('target_q', self.data.n_state_factors, n_hidden_units, self.data.n_actions)

        epsilon = tf.train.polynomial_decay(epsilon_start, tf.contrib.framework.get_global_step(),
                                            train_steps * (n_epochs - 1), end_learning_rate=epsilon_end,
                                            power=epsilon_decay)
        policy = make_epsilon_greedy_policy(q_estimator, epsilon, self.data.n_actions)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(n_epochs):
                print('\nEpoch {}'.format(epoch))
                print('  Global step: {}'.format(sess.run(tf.contrib.framework.get_global_step())))
                print('  Randomness: {}%'.format(int(100 * sess.run(epsilon))))

                state = self.data.start_training()
                losses = []

                # Fast-forward until all indicators have valid values
                while np.any(np.isnan(state)):
                    state = self.data.next()

                # Train the model
                for i in range(train_steps - self.data.train_counter):
                    action_probs = policy(sess, state)
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                    reward = self.data.step(action)
                    next_state = self.data.state()

                    states_batch = np.array([state])
                    action_batch = np.array([action])
                    reward_batch = np.array([reward])
                    next_states_batch = np.array([next_state])

                    q_values_next = q_estimator.predict(sess, next_states_batch)
                    targets_batch = reward_batch + gamma * np.amax(q_values_next, axis=1)

                    loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)
                    losses.append(loss)

                    state = self.data.state()

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
                    confidences.append((np.abs(q_values[0][0] - q_values[0][1])) / (np.abs(q_values[0][0]) + np.abs(q_values[0][1])))

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


    def run(self):
        pass
