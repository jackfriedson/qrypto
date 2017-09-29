import numpy as np

from qrypto.data.datasets import OHLCDataset


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
    actions = ['long', 'short']

    def __init__(self, *args, fee: float = 0.002, **kwargs):
        self.fee = fee

        self._normalized = False
        self.train_counter = None
        self.open_price = None
        self.position = 'long'
        self.test_position = 'long'

        super(QLearnDataset, self).__init__(*args, **kwargs)

    def start_training(self, start_step: int = 0):
        self.train_counter = start_step
        self._init_positions()
        self._init_orders()

        while np.any(np.isnan(self.state())):
            self.next()

        return self.train_counter

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
        result = self._data.iloc[self.last_idx]
        result.drop(EXCLUDE_FIELDS, inplace=True)
        result = result.values
        for indicator in self._indicators:
            row_vals = indicator.data.iloc[self.last_idx].values
            result = np.append(result, row_vals)
        return result

    @property
    def last(self):
        return self._data.iloc[self.last_idx]['close']

    @property
    def time(self):
        return self._data.iloc[self.last_idx].name

    @property
    def n_state_factors(self) -> int:
        return len(self.last_row) + 1

    @property
    def n_actions(self):
        return len(self.actions)

    def state(self):
        result = self.last_row

        if self._normalized:
            result = result - self.mean
            result = result / self.std

        result = np.append(result, 1. if self.position == 'long' else -1.)
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

    def step(self, idx: int):
        action = self.actions[idx]
        reward = 0.
        self.add_position(action)

        if self.position != action:
            reward -= self.fee

        self.position = action
        self.next()

        if self.position == 'long':
            reward += self.period_return
        else:
            reward -= self.period_return

        return reward

    def step_val(self, idx: int):
        action = self.actions[idx]
        test_reward = 0.

        if self.position != action:
            test_reward -= self.fee

            if action == 'long':
                self.add_order('buy', {'price': self.last})
            elif action == 'short':
                self.add_order('sell', {'price': self.last})

        train_reward = self.step(idx)

        if self.position == 'long':
            test_reward += self.period_return

        return train_reward, test_reward
