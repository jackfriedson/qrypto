import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cryptotrading.backtest import Backtest
from cryptotrading.data.datasets import QLearnDataset
from cryptotrading.data.indicators import BasicIndicator


log = logging.getLogger(__name__)


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
              validation_percent: float = 0.2):

        total_steps = len(self.exchange_train.date_range)
        train_steps = int((1. - validation_percent) * total_steps)
        validation_steps = int(validation_percent * total_steps) - 1

        tf.reset_default_graph()
        if random_seed:
            tf.set_random_seed(random_seed)
        global_step = tf.Variable(0, trainable=False)
        epsilon = tf.train.polynomial_decay(epsilon_start, global_step, train_steps * (n_epochs - 1),
                                            end_learning_rate=epsilon_end, power=epsilon_decay)

        n_inputs = self.data.n_state_factors
        n_hiddens = n_hidden_units
        n_outputs = self.data.n_actions

        input_layer = tf.placeholder(shape=[1, n_inputs], dtype=tf.float32)

        weights = {
            'hidden': tf.Variable(tf.random_normal([n_inputs, n_hiddens])),
            'output': tf.Variable(tf.random_normal([n_hiddens, n_outputs]))
        }

        hidden_layer = tf.matmul(input_layer, weights['hidden'])
        output_layer = tf.matmul(hidden_layer, weights['output'])
        predict = tf.argmax(output_layer, 1)

        targets = tf.placeholder(shape=[1, n_outputs], dtype=tf.float32)
        loss = tf.reduce_mean(tf.squared_difference(targets, output_layer))
        train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss, global_step=global_step)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(n_epochs):
                print('\nEpoch {}'.format(epoch))
                print('  Global step: {}'.format(sess.run(global_step)))
                print('  Randomness: {}%'.format(int(100 * sess.run(epsilon))))

                self.data.start_training()
                state = self.data.state()

                while np.any(np.isnan(state)):
                    # Fast-forward until all indicators have valid values
                    self.data.next()
                    state = self.data.state()

                # Train the model
                losses = []
                for i in range(train_steps - self.data.train_counter):
                    action, allQ = sess.run([predict, output_layer], feed_dict={input_layer: state})

                    if np.random.rand(1) < sess.run(epsilon):
                        # With probability epsilon, randomly try some other action
                        action[0] = np.random.randint(0, n_outputs)

                    reward = self.data.take_action_ls(action[0])
                    new_state = self.data.state()

                    Q_prime = sess.run(output_layer, feed_dict={input_layer: new_state})
                    Q_update = allQ
                    Q_update[0, action[0]] = reward + gamma * np.max(Q_prime)

                    _, l = sess.run([train_op, loss], feed_dict={input_layer: new_state, targets: Q_update})
                    losses.append(l)

                avg_loss = sum(losses) / len(losses)
                print('  Average loss: {}'.format(avg_loss))

                # Evaluate the model
                rewards = []
                returns = []
                confidences = []
                start_price = self.data.last

                for i in range(validation_steps):
                    state = self.data.state()
                    action, allQ = sess.run([predict, output_layer], feed_dict={input_layer: state})
                    reward, cum_return = self.data.test_action(action[0])
                    rewards.append(reward)
                    returns.append(cum_return)
                    confidences.append((np.abs(allQ[0][0] - allQ[0][1])) / (np.abs(allQ[0][0]) + np.abs(allQ[0][1])))

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
