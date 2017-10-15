from typing import Dict, Optional

import numpy as np
import pandas as pd

from qrypto.data.datasets import QLearnDataset


class CompositeQLearnDataset(object):
    actions = ['short', 'long']

    def __init__(self, primary_dataset: str, configs: Dict[str, list]) -> None:
        self._primary_name = primary_dataset
        self._datasets = {
            name: QLearnDataset(indicators=indicators) for name, indicators in configs.items()
        }

    @property
    def _primary(self):
        return self._datasets[self._primary_name]

    def start_training(self, start_step: int = 0):
        max_step = 0
        for dataset in self._datasets.values():
            max_step = max(max_step, dataset.start_training(start_step))
        for dataset in self._datasets.values():
            dataset.start_training(max_step)
        return max_step

    def next(self):
        for dataset in self._datasets.values():
            dataset.next()

    def stop_training(self):
        for dataset in self._datasets.values():
            dataset.stop_training()

    @property
    def n_state_factors(self):
        return len(self.last_row)

    @property
    def n_actions(self):
        return len(self.actions)

    def state(self):
        return self.last_row

    def step(self, idx: int):
        return self._primary.step(idx)

    def step_val(self, idx: int):
        return self._primary.step_val(idx)

    @property
    def all(self):
        result = pd.Dataframe()
        for pair, dataset in self._datasets.items():
            suffix = '_' + pair
            result.join(dataset.all(), rsuffix=suffix)
        return result

    @property
    def last_idx(self):
        return self._primary.last_idx

    @property
    def last_row(self):
        result = np.array([])
        for dataset in self._datasets.values():
            result = np.append(result, dataset.last_row)
        return result

    @property
    def last_price(self, pair: Optional[str] = None):
        dataset = self._datasets[pair] if pair else self._primary
        return dataset.last_price

    @property
    def time(self):
        return self._primary.time

    @property
    def period_return(self, pair: Optional[str] = None):
        dataset = self._datasets[pair] if pair else self._primary
        return dataset.period_return

    @property
    def cumulative_return(self, pair: Optional[str] = None):
        dataset = self._datasets[pair] if pair else self._primary
        return dataset.cumulative_return

    def init_data(self, data):
        for dataset in self._datasets.values():
            dataset.init_data(data)

    def update(self, incoming_data):
        for dataset in self._datasets.values():
            dataset.update(incoming_data)

    def add_position(self, *args):
        self._primary.add_position(*args)

    def add_order(self, *args):
        self._primary.add_order(*args)

    def plot(self, *args, **kwargs):
        self._primary.plot(*args, **kwargs)
