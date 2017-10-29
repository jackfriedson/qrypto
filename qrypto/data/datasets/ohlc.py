import io
from typing import List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from qrypto.types import OHLC


matplotlib.style.use('ggplot')


class OHLCDataset(object):
    """A dataset consisting of market candlestick data (open, high, low, close, and volume).
    Includes support for technical indicators, position and order tracking, and plotting.
    """

    def __init__(self,
                 data: Optional[List[OHLC]] = None,
                 indicators: Optional[List] = None,
                 interval: Optional[int] = None):
        self._data = None
        self._indicators = indicators or []
        self._interval = interval
        self._init_positions()
        self._init_orders()

        if data is not None:
            self.init_data(data)

        # TODO: Implement dynamic plotting of orders while running

    def init_data(self, data: List[OHLC]) -> None:
        self._data = pd.DataFrame(data)
        self._data.set_index('datetime', inplace=True)

        for indicator in self._indicators:
            indicator.update(self._data)

    def _init_positions(self) -> None:
        self._longs = {}
        self._shorts = {}

    def _init_orders(self) -> None:
        self._orders = pd.DataFrame(columns=['datetime', 'buy', 'sell'])
        self._orders.set_index('datetime', inplace=True)

    def update(self, incoming_data: List[OHLC]) -> None:
        for entry in incoming_data:
            datetime = entry.pop('datetime')
            self._data.loc[datetime] = entry

        for indicator in self._indicators:
            indicator.update(self._data)

    def add_position(self, long_short: str, time = None, price = None):
        price = price or self.last_price
        time = time or self.time
        if long_short == 'long':
            self._longs[time] = price
        elif long_short == 'short':
            self._shorts[time] = price

    def add_order(self, buy_sell: str, order_info: dict, time = None):
        time = time or self.time
        if buy_sell == 'buy':
            self._orders.loc[time, 'buy'] = order_info['price']
        elif buy_sell == 'sell':
            self._orders.loc[time, 'sell'] = order_info['price']

    def plot(self,
             data_column: str = 'close',
             plot_indicators: bool = False,
             plot_orders: bool = True,
             save_to: Union[str, io.BufferedIOBase] = None) -> None:
        fig = plt.figure(figsize=(60, 30))
        ratios = [3] if not plot_indicators else [3] + ([1] * len(self._indicators))
        n_subplots = 1 if not plot_indicators else 1 + len(self._indicators)
        gs = gridspec.GridSpec(n_subplots, 1, height_ratios=ratios)

        # Plot long and short positions
        ax0 = fig.add_subplot(gs[0])
        ax0.set_title('Price ({})'.format(data_column))
        ax0.plot(self._data.index, self._data[data_column], 'black')

        long_df = pd.DataFrame(np.nan, index=self._data.index, columns=[0])
        long_df.update(pd.DataFrame.from_dict(self._longs, orient='index'))
        long_df.plot(ax=ax0, style='g')

        short_df = pd.DataFrame(np.nan, index=self._data.index, columns=[0])
        short_df.update(pd.DataFrame.from_dict(self._shorts, orient='index'))
        short_df.plot(ax=ax0, style='r')

        if plot_orders:
            all_nan = self._orders.isnull().all(axis=0)
            if not all_nan['buy']:
                ax0.plot(self._orders.index, self._orders['buy'], color='k', marker='^', fillstyle='none')
            if not all_nan['sell']:
                ax0.plot(self._orders.index, self._orders['sell'], color='k', marker='v', fillstyle='none')

        if plot_indicators:
            for i, indicator in enumerate(self._indicators, start=1):
                ax_ind = fig.add_subplot(gs[i])
                indicator.plot(ax_ind)

        fig.autofmt_xdate()
        plt.tight_layout()

        if save_to:
            fig.savefig(save_to, format='png')
        else:
            plt.show()

    @property
    def all(self):
        result = self._data
        for indicator in self._indicators:
            result = result.join(indicator.data, rsuffix=indicator.suffix)
        return result

    @property
    def last_price(self):
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
