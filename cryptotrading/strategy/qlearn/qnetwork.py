import logging

import numpy as np
import tensorflow as tf

from cryptotrading.backtest import Backtest
from cryptotrading.data.datasets import QLearnDataset
from cryptotrading.data.indicators import BasicIndicator, MACD


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
        self.data = QLearnDataset(indicators)

    def update(self):
        new_data = self.exchange.get_ohlc(self.base_currency, self.quote_currency, interval=self.ohlc_interval)
        self.data.update(new_data)

    def train(self, learn_rate: float = 0.2, gamma: float = 0.95, n_epochs: int = 15):
        self.exchange = self.exchange_train

        crossval_pct = 0.2
        total_steps = len(self.exchange_train.date_range)
        train_steps = int((1. - crossval_pct) * total_steps)
        cv_steps = int(crossval_pct * total_steps)

        self.update()
        tf.reset_default_graph()
        inputs = tf.placeholder(shape=[1, self.data.n_state_factors], dtype=tf.float32)
        W = tf.Variable(tf.random_uniform([self.data.n_state_factors, self.data.n_actions], 0, 0.01))
        Qout = tf.matmul(inputs, W)
        predict = tf.argmax(Qout, 1)

        nextQ = tf.placeholder(shape=[1, self.data.n_actions], dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(nextQ - Qout))
        trainer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
        updateModel = trainer.minimize(loss)

        init = tf.global_variables_initializer()
        e = 0.3

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(n_epochs):
                self.reset_train_data()

                # Train the model
                for i in range(train_steps):
                    self.update()
                    state = self.data.state_vector()

                    if np.any(np.isnan(state)):
                        continue

                    action, allQ = sess.run([predict, Qout], feed_dict={inputs: state})

                    if np.random.rand(1) < e:
                        action[0] = np.random.randint(0, self.data.n_actions)

                    reward = self.data.take_action(action[0])
                    new_state = self.data.state_vector()
                    Q1 = sess.run(Qout, feed_dict={inputs:new_state})
                    maxQ1 = np.max(Q1)
                    targetQ = allQ
                    targetQ[0, action[0]] = reward + gamma * maxQ1
                    _, W1 = sess.run([updateModel, W], feed_dict={inputs: new_state, nextQ: targetQ})
                    # print(sess.run(loss, feed_dict={inputs: new_state, nextQ: targetQ}))

                rewards = []
                # Evaluate learning
                for i in range(cv_steps):
                    self.update()
                    state = self.data.state_vector()
                    action, _ = sess.run([predict, Qout], feed_dict={inputs: state})
                    reward = self.data.test_action(action[0])
                    if reward != 0.:
                        rewards.append(reward)

                e /= 2.
                avg_reward = sum(rewards) / (len(rewards) or 1.)
                print('Epoch {} -- average reward: {:.2f}%'.format(epoch, 100*avg_reward))

        self.data.plot()

    def reset_train_data(self):
        self.exchange_train.reset()
        self.data.reset()

    def run(self):
        self.exchange = self.exchange_run
        pass
