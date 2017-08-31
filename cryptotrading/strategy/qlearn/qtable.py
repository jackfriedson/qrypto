import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from cryptotrading.backtest import Backtest
from cryptotrading.data.datasets import QLearnDataset
from cryptotrading.data.indicators import BasicIndicator, MACD


log = logging.getLogger(__name__)


ACTIONS = ['do_nothing', 'buy_sell']


class QTableStrategy(object):

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

        indicators = [BasicIndicator('mom', {'timeperiod': 14})]
        self.data = QLearnDataset(indicators)

    def update(self):
        new_data = self.exchange.get_ohlc(self.base_currency, self.quote_currency, interval=self.ohlc_interval)
        self.data.update(new_data)

    def train(self, learn_rate: float = 0.3, gamma: float = 0.95, train_epochs: int = 10):
        self.exchange = self.exchange_train
        crossval_pct = 0.2

        # Initial update step lets us call n_states
        self.update()
        Q = np.zeros([self.data.n_states, len(ACTIONS)])
        total_steps = len(self.exchange_train.date_range)
        train_steps = int((1. - crossval_pct) * total_steps)
        cv_steps = int(crossval_pct * total_steps)

        n_epochs = 10

        self.reset_data()

        for epoch in range(n_epochs):
            # Train the model
            for i in range(train_steps):
                self.update()
                state = self.data.state

                if np.isnan(state):
                    continue

                action_idx = np.argmax(Q[state,:] + np.random.randn(1, len(ACTIONS)) * (1. / (i + 1)))
                action = ACTIONS[action_idx]
                new_state, reward = self.data.take_action(action)
                Q[state,action_idx] = Q[state,action_idx] + learn_rate * (reward + gamma * np.max(Q[new_state,:]) - Q[state,action_idx])

            rewards = []
            # Evaluate learning
            for i in range(cv_steps):
                self.update()
                state = self.data.state

                action_idx = np.argmax(Q[state,:])
                action = ACTIONS[action_idx]
                reward = self.data.test_action(action)
                if reward != 0.:
                    rewards.append(reward)

            avg_reward = sum(rewards) / (len(rewards) or 1.)
            print('Average reward: {:.2f}%'.format(100*avg_reward))
            self.reset_data()

    def reset_data(self):
        self.exchange_train.reset()
        self.data.reset()

    def run(self):
        self.exchange = self.exchange_run
        pass
