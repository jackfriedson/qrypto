import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec


matplotlib.style.use('ggplot')


class OHLCDataset(object):

    def __init__(self, data: List[dict] = None, indicators: List = None, charts_dir: str = None):
        self._data = None
        self._indicators = indicators or []
        self._init_positions()
        self._init_orders()

        if data is not None:
            self.init_data(data)

        self.charts_dir = charts_dir
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
        # TODO: Implement dynamic plotting of orders while running

    def init_data(self, data):
        self._data = None
        self.update(data)

    def _init_positions(self):
        self._positions = pd.DataFrame(columns=['datetime', 'long', 'short'])
        self._positions.set_index('datetime', inplace=True)

    def _init_orders(self):
        self._orders = pd.DataFrame(columns=['datetime', 'buy', 'sell'])
        self._orders.set_index('datetime', inplace=True)

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

    def add_position(self, long_short: str, order_info: dict):
        if long_short == 'long':
            self._positions.loc[self.time, 'long'] = order_info['price']
        elif long_short == 'short':
            self._positions.loc[self.time, 'short'] = order_info['price']

    def add_order(self, buy_sell: str, order_info: dict):
        if buy_sell == 'buy':
            self._orders.loc[self.time, 'buy'] = order_info['price']
        elif buy_sell == 'sell':
            self._orders.loc[self.time, 'sell'] = order_info['price']

    def plot(self, use_column: str = 'close', show: bool = True, filename: str = 'chart'):
        fig = plt.figure(figsize=(40, 30))
        ratios = [3] + ([1] * len(self._indicators))
        gs = gridspec.GridSpec(1 + len(self._indicators), 1, height_ratios=ratios)

        # Plot long and short positions
        ax0 = fig.add_subplot(gs[0])
        ax0.set_title('Price ({})'.format(use_column))
        ax0.plot(self._data.index, self._data[use_column], 'black')
        self._positions.plot(ax=ax0, style={'long': 'g', 'short': 'r'})
        all_nan = self._orders.isnull().all(axis=0)
        if not all_nan['buy']:
            ax0.plot(self._orders.index, self._orders['buy'], color='k', marker='^', fillstyle='none')
        if not all_nan['sell']:
            ax0.plot(self._orders.index, self._orders['sell'], color='k', marker='v', fillstyle='none')

        for i, indicator in enumerate(self._indicators, start=1):
            ax_ind = fig.add_subplot(gs[i])
            indicator.plot(ax_ind)

        fig.autofmt_xdate()
        plt.tight_layout()

        if self.charts_dir:
            fig.savefig(os.path.join(self.charts_dir, filename))

        if show:
            plt.show()

    @property
    def all(self):
        result = self._data
        for indicator in self._indicators:
            result = result.join(indicator.data, rsuffix=indicator.suffix)
        return result

    @property
    def last(self):
        return self._data.iloc[-1]['close']

    @property
    def time(self):
        return self._data.iloc[-1].name

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
