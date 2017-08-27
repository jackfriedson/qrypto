import numpy as np
import pandas as pd
import tensorflow as tf

from cryptotrading.backtest import Backtest
from cryptotrading.data.datasets import QLearnDataset


class QLearning(object):

    class _Dataset(QLearnDataset):
        pass

    def __init__(self, exchange, base_currency: str, quote_currency: str,
                 unit: float, ohlc_interval: int = 5, train_start: str = '6/1/2017',
                 train_end: str = '7/1/2017') -> None:
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.exchange = exchange
        self.exchange_train = Backtest(exchange, base_currency, quote_currency, train_start,
                                       train_end, ohlc_interval)
        self.unit = unit
        self.data = self._Dataset()

    def update(self):
        pass

    def train(self):
        pass

    def run(self):
        pass
