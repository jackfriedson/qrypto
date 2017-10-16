from typing import Dict, Optional

import numpy as np
import pandas as pd

from qrypto.data.datasets import QLearnDataset


class CompositeQLearnDataset(object):
    actions = ['short', 'long']

    def __init__(self, primary_name: str, configs: Dict[str, list]) -> None:
        self._primary_name = primary_name
        self._primary = QLearnDataset(indicators=configs.pop(primary_name))
        self._others = {
            name: QLearnDataset(indicators=indicators) for name, indicators in configs.items()
        }

    @property
    def _datasets(self):
        result = {
            self._primary_name: self._primary
        }
        result.update(self._others)
        return result

    def set_to(self, start_step: int = 0):
        max_step = 0

        for dataset in self._datasets.values():
            max_step = max(max_step, dataset.set_to(start_step))
        for dataset in self._datasets.values():
            dataset.set_to(max_step)

        return max_step

    def next(self):
        for dataset in self._datasets.values():
            dataset.next()

    @property
    def n_state_factors(self):
        return len(self.last_row)

    @property
    def n_actions(self):
        return len(self.actions)

    def state(self):
        return self.last_row

    def step(self, idx: int):
        for dataset in self._others.values():
            dataset.step(idx)
        return self._primary.step(idx)

    def validate(self, idx: int, place_orders: bool = True):
        for dataset in self._others.values():
            dataset.validate(idx, place_orders=False)
        return self._primary.validate(idx, place_orders=place_orders)

    @property
    def all(self):
        result = self._primary.all()
        for name, dataset in self._others.items():
            suffix = '_' + name
            result.join(dataset.all(), rsuffix=suffix)
        return result

    @property
    def last_idx(self):
        return self._primary.last_idx

    @property
    def last_row(self):
        result = self._primary.last_row
        for dataset in self._others.values():
            result = np.append(result, dataset.last_row)
        return result

    @property
    def last_price(self, name: Optional[str] = None):
        dataset = self._datasets[name] if name else self._primary
        return dataset.last_price

    @property
    def time(self):
        return self._primary.time

    @property
    def period_return(self, name: Optional[str] = None):
        dataset = self._datasets[name] if name else self._primary
        return dataset.period_return

    @property
    def cumulative_return(self, name: Optional[str] = None):
        dataset = self._datasets[name] if name else self._primary
        return dataset.cumulative_return

    def init_data(self, data, name: Optional[str] = None):
        dataset = self._datasets[name] if name else self._primary
        dataset.init_data(data)

    def update(self, data, name: Optional[str] = None):
        dataset = self._datasets[name] if name else self._primary
        dataset.update(data)

    def add_position(self, *args):
        self._primary.add_position(*args)

    def add_order(self, *args):
        self._primary.add_order(*args)

    def plot(self, *args, **kwargs):
        self._primary.plot(*args, **kwargs)
