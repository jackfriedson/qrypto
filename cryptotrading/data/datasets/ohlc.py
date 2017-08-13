from collections import OrderedDict

import numpy as np


class OHLCDataset(OrderedDict):

    def __init__(self, *args, **kwargs):
        """
        """
        self.max_size = kwargs.pop('max_size', 10000)
        self.last_timestamp = None
        super(OHLCDataset, self).__init__()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if len(self) >= self.max_size:
            self.popitem(last=False)

    def add_all(self, incoming_data):
        for entry in incoming_data:
            self[entry.get('time')] = entry
        self.last_timestamp = list(self.keys())[-1]

    @property
    def last(self):
        return self[self.last_timestamp].get('close')

    @property
    def open(self):
        return np.asarray([d['open'] for d in self.values()], dtype=float)

    @property
    def close(self):
        return np.asarray([d['close'] for d in self.values()], dtype=float)

    @property
    def high(self):
        return np.asarray([d['high'] for d in self.values()], dtype=float)

    @property
    def low(self):
        return np.asarray([d['low'] for d in self.values()], dtype=float)
