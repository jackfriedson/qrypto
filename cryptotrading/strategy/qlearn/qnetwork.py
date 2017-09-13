import logging
import os
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


summaries = os.path.expanduser('~/Desktop/tf_summaries')


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
        indicators.append(BasicIndicator('ppo', {'fastperiod': 10, 'slowperiod': 26, 'matype': 0}))
        self.data = QLearnDataset(indicators, **kwargs)
        self.data.update(self.exchange_train.all())
        self.data.normalize()

    def update(self):
        new_data = self.exchange.get_ohlc(self.base_currency, self.quote_currency, interval=self.ohlc_interval)
        self.data.update(new_data)

    def train(self,
              gamma: float = 0.95,
              n_epochs: int = 10,
              n_hidden_units: int = 3,
              random_seed: int = None,
              epsilon_start: float = .5,
              epsilon_end: float = 0.,
              epsilon_decay: float = 2,
              validation_percent: float = 0.2,
              replay_memory_start_size: int = 1000,
              replay_memory_max_size: int = 10000,
              replay_memory_batch_size: int = 16,
              update_target_every: int = 1000,
              save_chart_every: int = 1):

        total_steps = len(self.exchange_train.date_range)
        train_steps = int((1. - validation_percent) * total_steps)
        validation_steps = int(validation_percent * total_steps) - 1

        tf.reset_default_graph()

        if random_seed:
            tf.set_random_seed(random_seed)

        tf.Variable(0, name='global_step', trainable=False)

        q_estimator = QEstimator('q_estimator', self.data.n_state_factors, n_hidden_units,
                                 self.data.n_actions, summaries_dir=summaries)
        target_estimator = QEstimator('target_q', self.data.n_state_factors, n_hidden_units, self.data.n_actions)
        estimator_copy = ModelParametersCopier(q_estimator, target_estimator)

        epsilon = tf.train.polynomial_decay(epsilon_start, tf.contrib.framework.get_global_step(),
                                            train_steps * (n_epochs - 1), end_learning_rate=epsilon_end,
                                            power=epsilon_decay)
        policy = self.make_policy(q_estimator, epsilon, self.data.n_actions)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            replay_memory = deque(maxlen=replay_memory_max_size)
            nan_buffer = self.data.start_training()
            train_steps -= nan_buffer

            print('Initializing replay memory...')
            for i in range(min(replay_memory_start_size, train_steps)):
                state = self.data.state()
                action = policy(sess, state)
                reward = self.data.step(action)
                next_state = self.data.state()
                replay_memory.append(Transition(state, action, reward, next_state))

            for epoch in range(n_epochs):
                print('\nEpoch {}'.format(epoch))
                print('\tGlobal step: {}'.format(sess.run(tf.contrib.framework.get_global_step())))

                epoch_summary = tf.Summary()
                epoch_summary.value.add(simple_value=sess.run(epsilon), tag='epoch/train/epsilon')

                nan_buffer = self.data.start_training()
                train_rewards = []
                losses = []

                # Train the model
                for i in range(train_steps):

                    global_step = sess.run(tf.contrib.framework.get_global_step())
                    if (global_step // replay_memory_batch_size) % update_target_every == 0:
                        estimator_copy.make(sess)

                    state = self.data.state()
                    action = policy(sess, state)
                    reward = self.data.step(action)
                    next_state = self.data.state()

                    train_rewards.append(reward)

                    replay_memory.append(Transition(state, action, reward, next_state))

                    samples = random.sample(replay_memory, replay_memory_batch_size)
                    states_batch, action_batch, reward_batch, next_states_batch = map(np.array, zip(*samples))
                    # states_batch = np.array([state])
                    # action_batch = np.array([action])
                    # reward_batch = np.array([reward])
                    # next_states_batch = np.array([next_state])

                    q_values_next = target_estimator.predict(sess, next_states_batch)
                    targets_batch = reward_batch + gamma * np.amax(q_values_next, axis=1)

                    loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)
                    losses.append(loss)

                epoch_summary.value.add(simple_value=sum(train_rewards), tag='epoch/train/reward')
                epoch_summary.value.add(simple_value=np.average(losses), tag='epoch/train/averge_loss')

                # Evaluate the model
                rewards = []
                returns = []
                start_price = self.data.last

                for i in range(validation_steps):
                    state = self.data.state()
                    q_values = q_estimator.predict(sess, np.expand_dims(state, 0))[0]
                    #TODO: Calculate loss and plot against training loss
                    action = np.argmax(q_values)
                    reward, cum_return = self.data.step_val(action)

                    rewards.append(reward)
                    returns.append(cum_return)

                returns = list(filter(lambda x: x != 0, returns))

                position_value = start_price
                for return_val in returns:
                    position_value *= 1 + return_val
                market_return = (self.data.last / start_price) - 1.
                algorithm_return = (position_value / start_price) - 1.
                outperformance = algorithm_return - market_return

                print('\tMarket return: {:.2f}%'.format(100 * market_return))
                print('\tOutperformance: {:+.2f}%'.format(100 * outperformance))

                epoch_summary.value.add(simple_value=sum(rewards), tag='epoch/validate/reward')
                q_estimator.summary_writer.add_summary(epoch_summary, epoch)
                q_estimator.summary_writer.flush()

                if epoch % save_chart_every == 0 or epoch == n_epochs - 1:
                    self.data.plot(show=False, save_file='{}__{}.png'.format(int(time.time()), epoch))

    def make_policy(self, estimator, epsilon, n_actions):
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