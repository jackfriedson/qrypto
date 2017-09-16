import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec


matplotlib.style.use('ggplot')


class OHLCDataset(object):

    def __init__(self, data: List[dict] = None, indicators: List = None, charts_dir: str = None):
        self._data = None
        self._indicators = indicators or []
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

    def _init_orders(self):
        self._orders = pd.DataFrame(columns=['datetime', 'long', 'short'])
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

    def add_order(self, buy_sell: str, order_info: dict):
        if buy_sell in ['buy', 'long']:
            self._orders.loc[self.time, 'long'] = order_info['price']
        elif buy_sell in ['sell', 'short']:
            self._orders.loc[self.time, 'short'] = order_info['price']

    def plot(self, use_column: str = 'close', show: bool = True, filename: str = 'chart'):
        fig = plt.figure(figsize=(24, 18))
        ratios = [3] + ([1] * len(self._indicators))
        gs = gridspec.GridSpec(1 + len(self._indicators), 1, height_ratios=ratios)

        # Plot long and short positions
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(self._data.index, self._data[use_column], 'black')
        self._orders.plot(ax=ax0, style={'long': 'g', 'short': 'r'})
        ax0.set_title('Price ({})'.format(use_column))

        for i, indicator in enumerate(self._indicators, start=1):
            ax_ind = fig.add_subplot(gs[i])
            indicator.plot(ax_ind)

        fig.autofmt_xdate()
        plt.tight_layout()

        if self.charts_dir:
            fig.savefig(self.charts_dir + filename)

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
