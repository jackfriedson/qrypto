from typing import List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class OHLCDataset(object):

    def __init__(self, indicators: List = None):
        self._data = None
        self._indicators = indicators or []
        self._orders = {
            'buy': [],
            'sell': []
        }
        # TODO: Implement dynamic plotting of orders while running

    def __getattr__(self, name):
        for indicator in self._indicators:
            try:
                return getattr(indicator, name)
            except AttributeError:
                continue
        raise AttributeError

    def update(self, incoming_data: List[dict]) -> None:
        if self._data is None:
            self._data = pd.DataFrame(incoming_data)
            self._data.set_index('datetime', inplace=True)
        else:
            for entry in incoming_data:
                datetime = entry.pop('datetime')
                self._data.loc[datetime] = entry

        for indicator in self._indicators:
            indicator.update(self._data)

    def add_order(self, buy_sell: str, order_info: dict):
        self._orders[buy_sell].append((self.time, order_info['price']))

    def plot(self, column: str = 'close'):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self._data.index, self._data['close'])

        buy_dates, buy_prices = zip(*self._orders['buy'])
        sell_dates, sell_prices = zip(*self._orders['sell'])
        ax.plot(buy_dates, buy_prices, 'go')
        ax.plot(sell_dates, sell_prices, 'ro')

        # TODO: Plot indicators as subplots
        fig.autofmt_xdate()
        plt.show()

    @property
    def all(self):
        result = self._data
        for indicator in self._indicators:
            result = result.join(indicator.data)
        return result

    @property
    def last(self):
        return self._data.iloc[-1]['close']

    @property
    def time(self):
        return self._data.iloc[-1].name

    @property
    def open(self):
        return self._data['open'].values

    @property
    def close(self):
        return self._data['close'].values

    @property
    def high(self):
        return self._data['high'].values

    @property
    def low(self):
        return self._data['low'].values

    @property
    def volume(self):
        return self._data['volume'].values
