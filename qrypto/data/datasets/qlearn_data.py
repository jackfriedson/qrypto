from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from qrypto.data.datasets import CSVDataset, GKGDataset, OHLCDataset
from qrypto.types import OHLC


class QLearnDataset(object):
    """A dataset manager for use with machine learning algorithms."""
    actions = ['short', 'long']
    exclude_fields = ['open', 'high', 'low']

    def __init__(self,
                 ohlc_interval: int,
                 fee: float = 0.002,
                 indicators: Optional[list] = None,
                 csv_configs: Optional[Tuple[list, list]] = None,
                 gkg_file: Optional[Path] = None):
        self.fee = fee

        self._market_data = OHLCDataset(interval=ohlc_interval, indicators=indicators)
        self._csv_data = None
        self._gkg_data = None

        if csv_configs is not None:
            csv_files, custom_columns = csv_configs
            self._csv_data = CSVDataset(ohlc_interval, csv_files, custom_columns)

        if gkg_file is not None:
            self._gkg_data = GKGDataset(ohlc_interval, gkg_file)

        self._current_timestep = 0
        self._is_training = True
        self._train_data = None

        self._open_price = None
        self._position = 'long'

    def init_data(self, market_data):
        self._market_data.init_data(market_data)
        self._train_data = self.all.values
        self._columns = list(self.all.columns)

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
            self._market_data._init_positions()
            self._market_data._init_orders()

    def next(self):
        self._current_timestep += 1

    def update(self, new_data: List[OHLC]) -> None:
        self._market_data.update(new_data)

    def step(self, idx: int):
        action = self.actions[idx]
        reward = 0.
        self._market_data.add_position(action, self.time, self.last_price)

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
                self._market_data.add_order('buy', {'price': self.last_price}, self.time)
                self._open_price = self.last_price
            elif action == 'short':
                self._market_data.add_order('sell', {'price': self.last_price}, self.time)
                self._open_price = None

        train_reward = self.step(idx)

        if self._open_price is not None:
            test_reward += self.period_return

        return train_reward, test_reward

    def plot(self, **kwargs):
        self._market_data.plot(**kwargs)

    @property
    def all(self) -> pd.DataFrame:
        result = self._market_data.all
        result.drop(self.exclude_fields, axis=1, inplace=True)
        data_start = result.index[0]
        data_end = result.index[-1]

        if self._csv_data:
            blockchain_data = self._csv_data.between(data_start, data_end)
            result = result.join(blockchain_data)

        if self._gkg_data:
            gkg_data = self._gkg_data.between(data_start, data_end)
            result = result.join(gkg_data)

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
            result = self._market_data.last_row
            result.drop(self.exclude_fields, inplace=True)
            result = result.values
            for indicator in self._indicators:
                row_vals = indicator.data.iloc[self._last_idx].values
                result = np.append(result, row_vals)
            return result

    def get_last(self, column):
        idx = self._columns.index(column)
        return self.last_row[idx]

    @property
    def n_examples(self):
        return len(self._market_data._data)

    @property
    def last_price(self):
        return self.get_last('close')

    @property
    def time(self):
        return self._market_data._data.iloc[self._last_idx].name

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
        return (self._market_data.close[self._last_idx] / self._market_data.close[self._last_idx - 1]) - 1.

    @property
    def cumulative_return(self):
        if self._open_price:
            return (self.last_price / self._open_price) - 1.
        else:
            return 0.
