from typing import List, Optional

import numpy as np
import pandas as pd

from qrypto.data.datasets import OHLCDataset
from qrypto.types import OHLC


class QLearnDataset(object):
    """A dataset/environment manager for use with machine learning algorithms."""
    actions = ['short', 'long']
    exclude_fields = [
        'open',
        'high',
        'low',
    ]

    def __init__(self, fee: float = 0.002, indicators: Optional[list] = None):
        self.fee = fee

        self._ohlc_data = OHLCDataset(indicators=indicators)

        self._current_timestep = 0
        self._is_training = True
        self._train_data = None

        self._open_price = None
        self._position = 'long'

    def init_data(self, data):
        self._ohlc_data.init_data(data)
        self._train_data = self.all.values

    def set_training(self, is_training: bool):
        self._is_training = is_training

    def skip_nans(self) -> int:
        """Increments the current timestep until none of the values are NaN. Useful
        when using technical indicators that require at least n previous datapoints.
        """
        while np.any(np.isnan(self.state())):
            self.next()

        return self._current_timestep

    def set_to(self, start_step, reset_orders: bool = True) -> None:
        """Sets the current timestep to the given index."""
        self._current_timestep = start_step

        if reset_orders:
            self._ohlc_data._init_positions()
            self._ohlc_data._init_orders()

    def next(self):
        self._current_timestep += 1

    def update(self, new_data: List[OHLC]) -> None:
        self._ohlc_data.update(new_data)

    def step(self, idx: int):
        action = self.actions[idx]
        reward = 0.
        self._ohlc_data.add_position(action, self.time, self.last_price)

        if self._position != action:
            reward -= self.fee

        self._position = action
        self.next()

        if self._position == 'long':
            reward += self.period_return
        else:
            reward -= self.period_return

        return reward

    def validate(self, idx: int, place_orders: bool = True):
        action = self.actions[idx]
        test_reward = 0.

        if place_orders and self._position != action:
            test_reward -= self.fee

            if action == 'long':
                self._ohlc_data.add_order('buy', {'price': self.last_price}, self.time)
                self._open_price = self.last_price
            elif action == 'short':
                self._ohlc_data.add_order('sell', {'price': self.last_price}, self.time)
                self._open_price = None

        train_reward = self.step(idx)

        if self._open_price is not None:
            test_reward += self.period_return

        return train_reward, test_reward

    def plot(self, **kwargs):
        self._ohlc_data.plot(**kwargs)

    @property
    def all(self) -> pd.DataFrame:
        result = self._ohlc_data.all
        result.drop(self.exclude_fields, axis=1, inplace=True)
        return result

    @property
    def _last_idx(self):
        if self._is_training:
            return self._current_timestep
        else:
            return -1

    @property
    def last_row(self):
        if self._is_training:
            return self._train_data[self._current_timestep]
        else:
            result = self._ohlc_data.last_row
            result.drop(self.exclude_fields, inplace=True)
            result = result.values
            for indicator in self._indicators:
                row_vals = indicator.data.iloc[self._last_idx].values
                result = np.append(result, row_vals)
            return result

    @property
    def last_price(self):
        return self._ohlc_data._data.iloc[self._last_idx]['close']

    @property
    def time(self):
        return self._ohlc_data._data.iloc[self._last_idx].name

    @property
    def n_state_factors(self) -> int:
        return len(self.last_row)

    @property
    def n_actions(self):
        return len(self.actions)

    def state(self):
        result = self.last_row
        return result

    @property
    def period_return(self):
        return (self._ohlc_data.close[self._last_idx] / self._ohlc_data.close[self._last_idx - 1]) - 1.

    @property
    def cumulative_return(self):
        if self._open_price:
            return (self.last_price / self._open_price) - 1.
        else:
            return 0.
