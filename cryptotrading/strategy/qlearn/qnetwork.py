import logging

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
        self.exchange_run = exchange
        self.exchange_train = Backtest(exchange, base_currency, quote_currency, start=train_start,
                                       end=train_end, interval=ohlc_interval)
        self.exchange = None

        indicators = [BasicIndicator('mom', {'timeperiod': period}) for period in kwargs.pop('momentum')]
        indicators.append(BasicIndicator('mfi', {'timeperiod': 14}))
        indicators.append(BasicIndicator('natr', {'timeperiod': 14}))
        indicators.append(BasicIndicator('macd', {'fastperiod': 10, 'slowperiod': 26, 'signalperiod': 9}))
        indicators.append(BasicIndicator('bbands', {'timeperiod': 5, 'nbdevup': 2, 'nbdevdn': 2, 'matype': 0}))
        self.data = QLearnDataset(indicators, **kwargs)

    def update(self):
        new_data = self.exchange.get_ohlc(self.base_currency, self.quote_currency, interval=self.ohlc_interval)
        self.data.update(new_data)

    def train(self, learn_rate: float = 0.2, gamma: float = 0.98, n_epochs: int = 50):
        self.exchange = self.exchange_train

        crossval_pct = 0.2
        total_steps = len(self.exchange_train.date_range)
        train_steps = int((1. - crossval_pct) * total_steps)
        cv_steps = int(crossval_pct * total_steps)

        self.update()
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
        epsilon = tf.train.polynomial_decay(epsilon_start, global_step, train_steps * n_epochs,
                                            end_learning_rate=epsilon_end)

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(n_epochs):
                print('Epoch {}'.format(epoch))
                print('  Global step: {}'.format(sess.run(global_step)))
                print('  Randomness: {:.2f}'.format(sess.run(epsilon)))

                self.reset_train_data()
                self.update()
                state = self.data.state_vector()

                while np.any(np.isnan(state)):
                    # Fast-forward until all indicators have valid values
                    self.update()
                    state = self.data.state_vector()

                # Train the model
                # train_rewards = []
                for i in range(train_steps):
                    action, allQ = sess.run([predict, outputs], feed_dict={inputs: state})

                    if np.random.rand(1) < sess.run(epsilon):
                        # Randomly try some other action with probability e
                        action[0] = np.random.randint(0, self.data.n_actions)

                    reward = self.data.take_action_ls(action[0])
                    # if reward != 0.:
                    #     train_rewards.append(reward)

                    # consider splitting here
                    new_state = self.data.state_vector()
                    Q_prime = sess.run(outputs, feed_dict={inputs: new_state})
                    Q_update = allQ
                    Q_update[0, action[0]] = reward + gamma * np.max(Q_prime)

                    sess.run([train_op, W], feed_dict={inputs: new_state, targets: Q_update})

                # avg_train_reward = sum(train_rewards) / (len(train_rewards) or 1.)
                # print('  Average train reward: {:.2f}%'.format(100*avg_train_reward))
                print('  Loss: {}'.format(sess.run(loss, feed_dict={inputs: new_state, targets: Q_update})))

                # Evaluate learning
                rewards = []
                returns = []
                confidences = []
                for i in range(cv_steps):
                    self.update()
                    state = self.data.state_vector()
                    action, allQ = sess.run([predict, outputs], feed_dict={inputs: state})
                    rewards.append(self.data.take_action_ls(action[0]))
                    returns.append(self.data.test_action(action[0], add_order=False))
                    confidences.append((np.abs(allQ[0][0] - allQ[0][1])) / (np.abs(allQ[0][0]) + np.abs(allQ[0][1])))

                rewards = list(filter(lambda x: x != 0, rewards))
                returns = list(filter(lambda x: x != 0, returns))
                avg_reward = sum(rewards) / (len(rewards) or 1.)
                avg_return = sum(returns) / (len(returns) or 1.)
                avg_confidence = sum(confidences) / len(confidences)
                print('  Average reward: {:.2f}%'.format(100*avg_reward))
                print('  Average return: {:.2f}%'.format(100*avg_return))
                print('  Average confidence: {:.2f}%'.format(100*avg_confidence))

        self.data.plot()

    def reset_train_data(self):
        self.exchange_train.reset()
        self.data.reset()

    def run(self):
        self.exchange = self.exchange_run
        pass
