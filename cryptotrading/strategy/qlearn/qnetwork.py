import logging
import os
import random
import time
from collections import deque, namedtuple
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cryptotrading.backtest import Backtest
from cryptotrading.data.datasets import QLearnDataset
from cryptotrading.data.indicators import BasicIndicator
from cryptotrading.strategy.qlearn.qestimator import QEstimator, ModelParametersCopier


log = logging.getLogger(__name__)


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])


experiments_dir = os.path.expanduser('~/dev/cryptotrading/experiments/')
charts_dir = os.path.join(experiments_dir, 'charts')
summaries_dir = os.path.join(experiments_dir, 'tf_summaries')
models_dir = os.path.join(experiments_dir, 'models')


if not os.path.exists(charts_dir):
    os.makedirs(charts_dir)
if not os.path.exists(summaries_dir):
    os.makedirs(summaries_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


class QNetworkStrategy(object):

    def __init__(self,
                 exchange,
                 base_currency: str,
                 quote_currency: str,
                 unit: float,
                 ohlc_interval: int = 5,
                 sleep_duration: int = 5,
                 confidence_thresholds: Tuple[float, float] = (0.5, 0.5),
                 **kwargs) -> None:
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.unit = unit
        self.ohlc_interval = ohlc_interval
        self.confidence_thresholds = confidence_thresholds

        self.exchange = exchange
        self.timestamp = time.strftime('%Y%m%d_%H%M%S')

        self.charts_dir = os.path.join(charts_dir, self.timestamp)
        self.models_dir = os.path.join(models_dir, self.timestamp)

        indicators = []
        indicators.append(BasicIndicator('ppo', {'fastperiod': 10, 'slowperiod': 26, 'matype': 0}))
        self.data = QLearnDataset(indicators=indicators, charts_dir=self.charts_dir, **kwargs)

    def update(self):
        new_data = self.exchange.get_ohlc(self.base_currency, self.quote_currency, interval=self.ohlc_interval)
        self.data.update(new_data)

    def train(self,
              start: str,
              end: str,
              n_epochs: int = 10,
              validation_percent: float = 0.2,
              n_hidden_units: int = None,
              gamma: float = 0.95,
              epsilon_start: float = 0.5,
              epsilon_end: float = 0.,
              epsilon_decay: float = 2,
              experience_replay = True,
              replay_memory_start_size: int = 1000,
              replay_memory_max_size: int = 10000,
              replay_memory_batch_size: int = 16,
              update_target_every: int = 1000,
              random_seed: int = None,
              save_model: bool = True):

        # Initialize training data
        exchange_train = Backtest(self.exchange, self.base_currency, self.quote_currency,
                                  start=start, end=end, interval=self.ohlc_interval)
        self.data.init_data(exchange_train.all())
        self.data.normalize()

        # TODO: save training params to file for later reference

        total_steps = len(exchange_train.date_range)
        train_steps = int((1. - validation_percent) * total_steps)
        validation_steps = int(validation_percent * total_steps) - 1

        n_inputs = self.data.n_state_factors
        n_outputs = self.data.n_actions

        tf.reset_default_graph()

        if random_seed:
            tf.set_random_seed(random_seed)

        tf.Variable(0, name='global_step', trainable=False)

        q_estimator = QEstimator('q_estimator', n_inputs, n_outputs, hidden_units=n_hidden_units, summaries_dir=summaries_dir)
        target_estimator = QEstimator('target_q', n_inputs, n_outputs, hidden_units=n_hidden_units)
        estimator_copy = ModelParametersCopier(q_estimator, target_estimator)

        epsilon = tf.train.polynomial_decay(epsilon_start, tf.contrib.framework.get_global_step(),
                                            train_steps * (n_epochs - 1), end_learning_rate=epsilon_end,
                                            power=epsilon_decay)
        policy = self._make_policy(q_estimator, epsilon, n_outputs)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            nan_buffer = self.data.start_training()
            train_steps -= nan_buffer

            if experience_replay:
                print('Initializing replay memory...')
                replay_memory = deque(maxlen=replay_memory_max_size)
                for i in range(min(replay_memory_start_size, train_steps)):
                    state = self.data.state()
                    action = policy(sess, state)
                    reward = self.data.step(action)
                    next_state = self.data.state()
                    replay_memory.append(Transition(state, action, reward, next_state))

            for epoch in range(n_epochs):
                print('\nEpoch {}'.format(epoch))
                print('\tGlobal step: {}'.format(sess.run(tf.contrib.framework.get_global_step())))

                nan_buffer = self.data.start_training()
                train_rewards = []
                losses = []

                # Train the model
                for i in range(train_steps):
                    # Check whether to update target netowrk
                    global_step = sess.run(tf.contrib.framework.get_global_step())
                    if (global_step // replay_memory_batch_size) % update_target_every == 0:
                        estimator_copy.make(sess)

                    state = self.data.state()
                    action = policy(sess, state)
                    reward = self.data.step(action)
                    next_state = self.data.state()

                    if experience_replay:
                        replay_memory.append(Transition(state, action, reward, next_state))
                        samples = random.sample(replay_memory, replay_memory_batch_size)
                        states_batch, action_batch, reward_batch, next_states_batch = map(np.array, zip(*samples))
                    else:
                        states_batch = np.array([state])
                        action_batch = np.array([action])
                        reward_batch = np.array([reward])
                        next_states_batch = np.array([next_state])

                    # Update network
                    q_values_next, _ = target_estimator.predict(sess, next_states_batch)
                    targets_batch = reward_batch + gamma * np.amax(q_values_next, axis=1)
                    loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

                    train_rewards.append(reward)
                    losses.append(loss)

                if save_model:
                    saver.save(sess, os.path.join(self.models_dir, 'model.ckpt'))

                # Evaluate the model
                rewards = []
                returns = []
                val_losses = []
                start_price = self.data.last

                for i in range(validation_steps):
                    state = self.data.state()
                    q_values, confidence_values = q_estimator.predict(sess, np.expand_dims(state, 0))
                    action = np.argmax(q_values)
                    confidence = confidence_values[0][action]
                    reward, cum_return = self.data.step_val(action, confidence, self.confidence_thresholds)

                    # Calculate validation loss for summaries
                    next_state = self.data.state()
                    next_q_values = q_estimator.predict(sess, np.expand_dims(next_state, 0))[0]
                    target = reward + gamma * np.amax(next_q_values)
                    loss = q_estimator.compute_loss(sess, np.array([state]), np.array([action]), np.array([target]))

                    rewards.append(reward)
                    returns.append(cum_return)
                    val_losses.append(loss)

                # Compute outperformance of market return
                position_value = start_price
                for return_val in returns:
                    position_value *= 1 + return_val
                market_return = (self.data.last / start_price) - 1.
                algorithm_return = (position_value / start_price) - 1.
                outperformance = algorithm_return - market_return
                print('\tMarket return: {:.2f}%'.format(100 * market_return))
                print('\tOutperformance: {:+.2f}%'.format(100 * outperformance))

                # Add Tensorboard summaries
                epoch_summary = tf.Summary()
                epoch_summary.value.add(simple_value=sess.run(epsilon), tag='epoch/train/epsilon')
                epoch_summary.value.add(simple_value=sum(train_rewards), tag='epoch/train/reward')
                epoch_summary.value.add(simple_value=np.average(losses), tag='epoch/train/averge_loss')
                epoch_summary.value.add(simple_value=sum(rewards), tag='epoch/validate/reward')
                epoch_summary.value.add(simple_value=np.average(val_losses), tag='epoch/validate/average_loss')
                q_estimator.summary_writer.add_summary(epoch_summary, epoch)
                q_estimator.summary_writer.flush()

                # Save 10 graphs per training session
                if epoch in range(0, n_epochs, n_epochs // 10):
                    self.data.plot(show=False, filename='epoch_{}'.format(epoch))

    @staticmethod
    def _make_policy(estimator, epsilon, n_actions):
        def policy_fn(sess, observation):
            epsilon_val = sess.run(epsilon)
            action_probs = np.ones(n_actions, dtype=float) * epsilon_val / n_actions
            q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
            best_action = np.argmax(q_values)
            action_probs[best_action] += (1.0 - epsilon_val)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            return action
        return policy_fn

    def _scatter_plot(self, x, y, filename):
        fig = plt.figure(figsize=(12,9))
        plt.scatter(x, y)
        fig.savefig(self.charts_dir + filename)

    def run(self):
        pass
