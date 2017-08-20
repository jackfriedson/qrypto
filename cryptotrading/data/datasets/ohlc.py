from typing import List

import numpy as np
import pandas as pd


class OHLCDataset(object):

    def __init__(self, *args, **kwargs):
        self._data = None
        super(OHLCDataset, self).__init__(*args, **kwargs)

    def add(self, entry: dict) -> None:
        datetime = entry.pop('datetime')
        self._data.loc[datetime] = entry

    def update(self, incoming_data: List[dict]) -> None:
        if self._data is None:
            self._data = pd.DataFrame(incoming_data)
            self._data.set_index('datetime', inplace=True)
        else:
            for entry in incoming_data:
                self.add(entry)

        try:
            super(OHLCDataset, self).update(incoming_data)
        except AttributeError:
            pass

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
