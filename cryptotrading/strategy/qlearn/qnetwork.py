import io
import logging
import os
import random
import time
from collections import deque, namedtuple
from typing import Tuple

import numpy as np
import tensorflow as tf

from cryptotrading.backtest import Backtest
from cryptotrading.data.datasets import QLearnDataset
from cryptotrading.data.indicators import BasicIndicator, DifferenceIndicator
from cryptotrading.strategy.qlearn.experience_buffer import ExperienceBuffer
from cryptotrading.strategy.qlearn.qestimator import QEstimator, ModelParametersCopier


log = logging.getLogger(__name__)


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])


experiments_dir = os.path.expanduser('~/dev/cryptotrading/experiments/')
summaries_dir = os.path.join(experiments_dir, 'tf_summaries')
models_dir = os.path.join(experiments_dir, 'models')


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
        self.models_dir = os.path.join(models_dir, self.timestamp)

        indicators = [
            BasicIndicator('ppo'),
        ]
        self.data = QLearnDataset(indicators=indicators, **kwargs)

    def update(self):
        new_data = self.exchange.get_ohlc(self.base_currency, self.quote_currency, interval=self.ohlc_interval)
        self.data.update(new_data)

    def train(self,
              start: str,
              end: str,
              n_epochs: int = 10,
              validation_percent: float = 0.2,
              gamma: float = 0.95,
              epsilon_start: float = 1.,
              epsilon_end: float = 0.,
              epsilon_decay: float = 2,
              random_action_freq: int = 4,
              replay_memory_start_size: int = 1000,
              replay_memory_max_size: int = 40000,
              batch_size: int = 4,
              trace_length: int = 8,
              update_target_every: int = 1000,
              random_seed: int = None,
              save_model: bool = True):

        # Initialize training data
        exchange_train = Backtest(self.exchange, self.base_currency, self.quote_currency,
                                  start=start, end=end, interval=self.ohlc_interval)
        self.data.init_data(exchange_train.all())
        n_inputs = self.data.n_state_factors
        n_outputs = self.data.n_actions
        n_hiddens = (n_inputs + n_outputs) // 2
        random = np.random.RandomState(random_seed)

        # TODO: save training params to file for later reference

        total_steps = len(exchange_train.date_range)
        nan_buffer = self.data.start_training()
        total_steps -= nan_buffer

        epoch_step_ratio = 1. / (1. + ((n_epochs - 1) * validation_percent))
        epoch_steps = int(epoch_step_ratio * total_steps)
        train_steps = int(epoch_steps * (1. - validation_percent))
        validation_steps = int(epoch_steps * validation_percent)
        initial_step = nan_buffer

        tf.reset_default_graph()
        global_step = tf.Variable(0, name='global_step', trainable=False)

        if random_seed:
            tf.set_random_seed(random_seed)

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_inputs, state_is_tuple=True)
        cellT = tf.contrib.rnn.BasicLSTMCell(num_units=n_inputs, state_is_tuple=True)
        q_estimator = QEstimator('q_estimator', cell, n_inputs, n_hiddens, n_outputs, summaries_dir=summaries_dir)
        target_estimator = QEstimator('target_q', cellT, n_inputs, n_hiddens, n_outputs)
        estimator_copy = ModelParametersCopier(q_estimator, target_estimator)

        epsilon = tf.train.polynomial_decay(epsilon_start, global_step,
                                            train_steps * (n_epochs - 1), end_learning_rate=epsilon_end,
                                            power=epsilon_decay)
        policy = self._make_policy(q_estimator, epsilon, n_outputs, random_action_freq)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            rnn_state = (np.zeros([1, n_inputs]), np.zeros([1, n_inputs]))

            print('Initializing replay memory...')
            replay_memory = ExperienceBuffer(replay_memory_max_size, random=random)

            for _ in range(min(replay_memory_start_size, train_steps)):
                state = self.data.state()
                action = random.randint(n_outputs)
                reward = self.data.step(action)
                next_state = self.data.state()
                replay_memory.add(Transition(state, action, reward, next_state))

            for epoch in range(n_epochs):
                self.data.start_training(initial_step)
                initial_step += validation_steps
                replay_memory.new_episode()
                rnn_state = (np.zeros([1, n_inputs]), np.zeros([1, n_inputs]))

                print('\nEpoch {}'.format(epoch))
                print('\tGlobal step: {}'.format(sess.run(global_step)))
                print('\tStarting at: {}'.format(self.data.time))

                train_rewards = []
                losses = []

                for _ in range(train_steps):
                    # Maybe update the target network
                    if (sess.run(global_step) // batch_size) % update_target_every == 0:
                        estimator_copy.make(sess)

                    # Make a prediction
                    state = self.data.state()
                    action, next_rnn_state = policy(sess, state, rnn_state)
                    reward = self.data.step(action)
                    next_state = self.data.state()

                    replay_memory.add(Transition(state, action, reward, next_state))
                    samples = replay_memory.sample(batch_size, trace_length)
                    states_batch, action_batch, reward_batch, next_states_batch = map(np.array, zip(*samples))

                    # Train the network
                    train_rnn_state = (np.zeros([batch_size, n_inputs]), np.zeros([batch_size, n_inputs]))
                    q_values_next = target_estimator.predict(sess, next_states_batch, trace_length, train_rnn_state)[0]
                    targets_batch = reward_batch + gamma * np.amax(q_values_next, axis=1)
                    loss = q_estimator.update(sess, states_batch, action_batch, targets_batch, trace_length, train_rnn_state)

                    rnn_state = next_rnn_state

                    train_rewards.append(reward)
                    losses.append(loss)

                if save_model:
                    saver.save(sess, os.path.join(self.models_dir, 'model.ckpt'))

                # Evaluate the model
                rewards = []
                returns = []
                confidences = []
                val_losses = []
                start_price = self.data.last

                rnn_state = (np.zeros([1, n_inputs]), np.zeros([1, n_inputs]))

                for _ in range(validation_steps):
                    state = self.data.state()
                    q_values, confidence, next_rnn_state = q_estimator.predict(sess, np.expand_dims(state, 0), 1, rnn_state, training=False)
                    action = np.argmax(q_values)
                    confidence = confidence[0][action]
                    reward, cum_return = self.data.step_val(action, confidence, self.confidence_thresholds)

                    # Calculate validation loss for summaries
                    next_state = self.data.state()
                    next_q_values = q_estimator.predict(sess, np.expand_dims(next_state, 0), 1, next_rnn_state, training=False)[0]
                    target = reward + gamma * np.amax(next_q_values)
                    loss = q_estimator.compute_loss(sess, state, action, target, rnn_state)

                    rnn_state = next_rnn_state

                    rewards.append(reward)
                    returns.append(cum_return)
                    confidences.append(confidence)
                    val_losses.append(loss)

                # Compute outperformance of market return
                position_value = start_price
                market_return = (self.data.last / start_price) - 1.

                returns = list(filter(None, returns))
                if returns:
                    for return_val in returns:
                        position_value *= 1 + return_val
                    algorithm_return = (position_value / start_price) - 1.
                else:
                    # No buys/sells were made
                    algorithm_return = market_return if self.data.position == 'long' else 0.
                outperformance = algorithm_return - market_return
                print('\tMarket return: {:.2f}%'.format(100 * market_return))
                print('\tOutperformance: {:+.2f}%'.format(100 * outperformance))

                buf = io.BytesIO()
                self.data.plot(save_to=buf)
                buf.seek(0)
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                image = tf.expand_dims(image, 0)
                epoch_chart = tf.summary.image('epoch_{}'.format(epoch), image, max_outputs=1).eval()

                # Add Tensorboard summaries
                epoch_summary = tf.Summary()
                epoch_summary.value.add(simple_value=sess.run(epsilon), tag='epoch/train/epsilon')
                epoch_summary.value.add(simple_value=sum(train_rewards), tag='epoch/train/reward')
                epoch_summary.value.add(simple_value=np.average(losses), tag='epoch/train/averge_loss')
                epoch_summary.value.add(simple_value=sum(rewards), tag='epoch/validate/reward')
                epoch_summary.value.add(simple_value=outperformance, tag='epoch/validate/outperformance')
                epoch_summary.value.add(simple_value=np.average(confidences), tag='epoch/validate/average_confidence')
                epoch_summary.value.add(simple_value=np.average(val_losses), tag='epoch/validate/average_loss')
                q_estimator.summary_writer.add_summary(epoch_summary, epoch)
                q_estimator.summary_writer.add_summary(epoch_chart, epoch)
                q_estimator.summary_writer.flush()

    @staticmethod
    def _make_policy(estimator, epsilon, n_actions, random_action_freq):
        def policy_fn(sess, observation, rnn_state):
            epsilon_val = sess.run(epsilon)
            step = sess.run(tf.contrib.framework.get_global_step())
            q_values, _, new_rnn_state = estimator.predict(sess, np.expand_dims(observation, 0), 1, rnn_state)
            best_action = np.argmax(q_values)
            if step % random_action_freq == 0:
                action_probs = np.ones(n_actions, dtype=float) * epsilon_val / n_actions
                action_probs[best_action] += (1.0 - epsilon_val)
                return random.choice(np.arange(len(action_probs)), p=action_probs), new_rnn_state
            else:
                return best_action, new_rnn_state
        return policy_fn

    def run(self):
        pass
