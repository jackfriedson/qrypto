import logging
import time

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

    def train(self, learn_rate: float = 0.2, gamma: float = 0.98, n_epochs: int = 20):
        # TODO: Group all parameters and meta-parameters in one place

        val_percent = 0.2
        total_steps = len(self.exchange_train.date_range)
        train_steps = int((1. - val_percent) * total_steps)
        val_steps = int(val_percent * total_steps)

        tf.reset_default_graph()
        global_step = tf.Variable(0, trainable=False)
        inputs = tf.placeholder(shape=[1, self.data.n_state_factors], dtype=tf.float32)
        W = tf.Variable(tf.random_uniform([self.data.n_state_factors, self.data.n_actions], 0, 0.01))
        outputs = tf.matmul(inputs, W)
        predict = tf.argmax(outputs, 1)

        targets = tf.placeholder(shape=[1, self.data.n_actions], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(targets - outputs))
        train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss, global_step=global_step)
        init = tf.global_variables_initializer()

        epsilon_start = 0.5
        epsilon_end = 0.
        decay_power = 1.5
        epsilon = tf.train.polynomial_decay(epsilon_start, global_step, train_steps * (n_epochs - 1),
                                            end_learning_rate=epsilon_end, power=decay_power)

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(n_epochs):
                print('Epoch {}'.format(epoch))
                print('  Global step: {}'.format(sess.run(global_step)))
                print('  Randomness: {:.2f}'.format(sess.run(epsilon)))

                self.data.start_training()
                state = self.data.state()

                while np.any(np.isnan(state)):
                    # Fast-forward until all indicators have valid values
                    self.data.next()
                    state = self.data.state()

                # Train the model
                for i in range(train_steps - self.data.train_counter):
                    action, allQ = sess.run([predict, outputs], feed_dict={inputs: state})

                    if np.random.rand(1) < sess.run(epsilon):
                        # Randomly try some other action with probability e
                        action[0] = np.random.randint(0, self.data.n_actions)

                    reward = self.data.take_action_ls(action[0])
                    new_state = self.data.state()

                    Q_prime = sess.run(outputs, feed_dict={inputs: new_state})
                    Q_update = allQ
                    Q_update[0, action[0]] = reward + gamma * np.max(Q_prime)

                    sess.run([train_op, W], feed_dict={inputs: new_state, targets: Q_update})

                print('  Loss: {}'.format(sess.run(loss, feed_dict={inputs: new_state, targets: Q_update})))

                # Evaluate learning
                rewards = []
                returns = []
                confidences = []
                for i in range(val_steps):
                    state = self.data.state()
                    action, allQ = sess.run([predict, outputs], feed_dict={inputs: state})
                    reward, cum_return = self.data.test_action(action[0])
                    rewards.append(reward)
                    returns.append(cum_return)
                    confidences.append((np.abs(allQ[0][0] - allQ[0][1])) / (np.abs(allQ[0][0]) + np.abs(allQ[0][1])))

                rewards = list(filter(lambda x: x != 0, rewards))
                returns = list(filter(lambda x: x != 0, returns))
                avg_reward = sum(rewards) / (len(rewards) or 1.)
                avg_return = sum(returns) / (len(returns) or 1.)
                avg_confidence = sum(confidences) / len(confidences)
                print('  Average reward: {:.2f}%'.format(100*avg_reward))
                print('  Average return: {:.2f}%'.format(100*avg_return))
                print('  Average confidence: {:.2f}%'.format(100*avg_confidence))

                if epoch in [0, n_epochs // 2, n_epochs - 1]:
                    self.data.plot(show=False, save_file='{}__{}.png'.format(int(time.time()), epoch))

    def run(self):
        pass
