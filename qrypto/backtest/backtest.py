import sys
from pathlib import Path
from typing import Optional

import pandas as pd


class Backtest(object):
    """ Wrapper around an exchange's API adapter in order to provide an easy interface for
    backtesting.
    """

    def __init__(self,
                 exchange,
                 base_currency: str,
                 quote_currency: str,
                 start: str = '1/1/2016',
                 end: str = '1/1/2017',
                 interval: int = 5,
                 save_to_csv: Optional[Path] = None):
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.date_range = pd.date_range(self.start, self.end, freq='%dMin' % interval)
        self.interval = interval

        # TODO: Verify this works with Kraken API as well
        unix_start = self.start.value // 10**9
        unix_end = self.end.value // 10**9
        all_data = exchange.get_ohlc(self.base_currency, self.quote_currency, self.interval,
                                     start=unix_start, end=unix_end)
        self._test_data = pd.DataFrame(all_data)
        self._test_data.set_index('datetime', inplace=True)

        if save_to_csv:
            filename = base_currency.lower() + '_ohlc.csv'
            full_path = save_to_csv/filename
            self._test_data.to_csv(full_path.resolve())

        self.call_count = 0
        self.open_price = None
        self.orders = []

    def reset(self):
        self.call_count = 0

    def print_results(self) -> None:
        n_orders = len(self.orders)
        total_pl = sum(map(lambda o: o['profit_loss'] * (1. / n_orders), self.orders))
        for order in self.orders:
            print(order)
        print('Placed {} orders. Total profit/loss: {:.2f}%'.format(n_orders, total_pl))

    def get_ohlc(self, *args, **kwargs) -> list:
        if self.call_count >= len(self.date_range):
            raise StopIteration('No more test data')

        current_datetime = self.date_range[self.call_count]
        row = self._test_data.loc[current_datetime]
        result = row.to_dict()
        result['datetime'] = current_datetime
        self.call_count += 1
        return [result]

    def all(self):
        def _datetime_to_ohlc_dict(date):
            row = self._test_data.loc[date]
            row_dict = row.to_dict()
            row_dict['datetime'] = date
            return row_dict
        return [_datetime_to_ohlc_dict(date) for date in self.date_range]

    def limit_order(self, base_currency: str, buy_sell: str, price: float, volume: float, **kwargs) -> dict:
        if buy_sell == 'buy':
            self.open_price = price
        else:
            self.orders.append({
                'open': self.open_price,
                'close': price,
                'volume': volume,
                'profit_loss': (100. * ((price / self.open_price) - 1.))
            })
            self.open_price = None

        return {
            'price': price,
            'fee': 0.00
        }