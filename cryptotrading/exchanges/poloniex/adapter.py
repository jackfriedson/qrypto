from typing import Callable, Optional

import pandas as pd
from src import PoloniexAPI

class PoloniexAdapter(object):

    def __init__(self, key_path: str) -> None:
        apikey, secret = self.load_api_key(key_path)
        self.api = PoloniexAPI(apikey, secret)

    @classmethod
    def load_api_key(cls, path: str) -> dict:
        with open(path, 'rb') as f:
            key = f.readline().strip()
            secret = f.readline().strip()
        return key, secret

    @classmethod
    def pair(cls, base_currency: str, quote_currency: str) -> str:
        return quote_currency + '_' + base_currency

    def subscribe(self, callback: Callable) -> None:
        self.api.subscribe('ticker', callback)

    def get_ohlc(self,
                 base_currency: str,
                 quote_currency: str = 'USDT',
                 interval: int = 5,
                 since: Optional[int] = None) -> dict:
        args = {
            'currencyPair': self.pair(base_currency, quote_currency),
            'period': interval * 60
        }
        if since:
            args.update({'start': since})

        result = self.api.returnChartData(**args)

        def format_ohlc(datapoint):
            datapoint['datetime'] = pd.to_datetime(datapoint.pop('date'), unit='s')
            datapoint['avg'] = datapoint.pop('weightedAverage')
            return datapoint

        return list(map(format_ohlc, result))

    def get_balance(self) -> dict:
        result = self.api.returnBalances()
        return result

    def market_order(self, base_currency, buy_sell, volume, **kwargs):
        print('POLONIEX: Place order to {} {} {}'.format(buy_sell, volume, base_currency))
