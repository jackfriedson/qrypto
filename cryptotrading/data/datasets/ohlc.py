from typing import List

import numpy as np
import pandas as pd


class OHLCDataset(object):

    def __init__(self, *args, **kwargs):
        # self.max_size = kwargs.pop('max_size', 10000)
        self._data = None

    def add(self, entry: dict):
        datetime = entry.pop('datetime')
        self._data.loc[datetime] = entry

    def add_all(self, incoming_data: List[dict]):
        if self._data is None:
            self._data = pd.DataFrame(incoming_data)
            self._data.set_index('datetime', inplace=True)
        else:
            for entry in incoming_data:
                self.add(entry)

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
