from typing import List

import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.style.use('ggplot')


class OHLCDataset(object):

    def __init__(self, indicators: List = None):
        self._data = None
        self._indicators = indicators or []
        self._orders = {
            'buy': [],
            'sell': []
        }
        # TODO: Implement dynamic plotting of orders while running

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
        self._orders[buy_sell].append((self.time, order_info['price']))

    def plot(self, use_column: str = 'close'):
        fig = plt.figure(figsize=(12, 9))
        ratios = [3] + ([1] * len(self._indicators))
        gs = gridspec.GridSpec(1 + len(self._indicators), 1, height_ratios=ratios)
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(self._data.index, self._data[use_column])
        ax0.set_title('Price (' + use_column.title() + ')')

        buy_dates, buy_prices = zip(*self._orders['buy'])
        sell_dates, sell_prices = zip(*self._orders['sell'])
        ax0.plot(buy_dates, buy_prices, 'g.')
        ax0.plot(sell_dates, sell_prices, 'r.')

        for i, indicator in enumerate(self._indicators, start=1):
            ax_ind = fig.add_subplot(gs[i], sharex=ax0)
            indicator.plot(ax_ind)

        fig.autofmt_xdate()
        plt.tight_layout()
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
