from typing import Tuple

import numpy as np

from cryptotrading.data.datasets import OHLCDataset


EXCLUDE_FIELDS = [
    'open',
    'high',
    'low',
    'close',
    'volume',
    'quoteVolume',
    'avg'
]


class QLearnDataset(OHLCDataset):
    actions = ['buy', 'sell']

    def __init__(self, *args, fee: float = 0.002, **kwargs):
        self.fee = fee

        self._normalized = False
        self.train_counter = None
        self.open_price = None
        self.position = 'long'

        super(QLearnDataset, self).__init__(*args, **kwargs)

    def start_training(self):
        self.train_counter = 0
        self._orders['buy'] = []
        self._orders['sell'] = []

    def next(self):
        self.train_counter += 1

    def stop_training(self):
        self.train_counter = None

    def normalize(self):
        self._normalized = True
        self.mean = self.all.mean()
        self.std = self.all.std()

    @property
    def all(self):
        result = super(QLearnDataset, self).all
        result.drop(EXCLUDE_FIELDS, axis=1, inplace=True)
        return result

    @property
    def last_idx(self):
        return self.train_counter if self.train_counter is not None else -1

    @property
    def last_row(self):
        return self.all.iloc[self.last_idx]

    @property
    def last(self):
        return self._data.iloc[self.last_idx]['close']

    @property
    def time(self):
        return self._data.iloc[self.last_idx].name

    @property
    def n_state_factors(self) -> int:
        return len(self.last_row)

    @property
    def n_actions(self):
        return len(self.actions)

    def state(self):
        result = self.last_row

        if self._normalized:
            result = result - self.mean
            result = result / self.std

        result = result.values.reshape(1, len(result))
        return result

    @property
    def period_return(self):
        return (self.close[self.last_idx] / self.close[self.last_idx - 1]) - 1.

    @property
    def cumulative_return(self):
        if self.open_price:
            return (self.last / self.open_price) - 1.
        else:
            return 0.

    def take_action_ls(self, idx: int):
        action = self.actions[idx]
        self.add_order(action, {'price': self.last})

        if action == 'buy':
            self.position = 'long'
        else:
            self.position = 'short'

        self.next()

        if self.position == 'long':
            return self.period_return
        else:
            return -self.period_return

    def test_action(self, idx: int):
        action = self.actions[idx]
        if action == 'buy' and not self.open_price:
            self.open_price = self.last
            cum_return = 0.
        elif action == 'sell' and self.open_price:
            cum_return = self.cumulative_return - (2 * self.fee)
            self.open_price = None
        else:
            cum_return = 0.

        reward = self.take_action_ls(idx)

        return reward, cum_return
