from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from cryptotrading.data.indicators import BaseIndicator, BasicIndicator


class OHLCDataset(object):

    def __init__(self, indicators: List = None):
        self._data = None
        self._indicators = indicators or []

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

    def plot(self, column='close'):
        self._data[column].plot()
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
