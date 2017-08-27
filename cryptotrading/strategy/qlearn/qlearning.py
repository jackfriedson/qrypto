import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from cryptotrading.backtest import Backtest
from cryptotrading.data.datasets import QLearnDataset
from cryptotrading.data.mixins import MACDMixin, MFIMixin, RSIMixin


log = logging.getLogger(__name__)


class QLearnStrategy(object):

    class _Dataset(QLearnDataset, MFIMixin, RSIMixin):
        pass

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
        self.data = self._Dataset(**kwargs)

    def update(self):
        new_data = self.exchange.get_ohlc(self.base_currency, self.quote_currency, interval=self.ohlc_interval)
        self.data.update(new_data)

    def train(self):
        self.exchange = self.exchange_train
        n_iters = len(self.exchange_train.date_range)

        for _ in range(n_iters):
            self.update()
            print(self.data.state)

    def run(self):
        self.exchange = self.exchange_run
        pass
