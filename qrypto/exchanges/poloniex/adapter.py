import time
from typing import Callable, Optional

import pandas as pd
from poloniex import PoloniexAPI

from qrypto.exchanges import BaseAPIAdapter


class PoloniexAPIAdapter(BaseAPIAdapter):

    def __init__(self, key_path: str) -> None:
        apikey, secret = self.load_api_key(key_path)
        self.api = PoloniexAPI(apikey, secret)
        self.last = {}

    @classmethod
    def load_api_key(cls, path: str) -> dict:
        with open(path, 'rb') as f:
            key = f.readline().strip()
            secret = f.readline().strip()
        return key, secret

    @staticmethod
    def currency_pair(base_currency: str, quote_currency: str) -> str:
        # Poloniex only has Tether exchanges, not USD
        if quote_currency == 'USD':
            quote_currency = 'USDT'

        return quote_currency + '_' + base_currency

    def subscribe(self, callback: Callable) -> None:
        self.api.subscribe('ticker', callback)

    def get_ohlc(self, base_currency: str, quote_currency: str = 'USDT', interval: int = 5,
                 since_last: bool = False, **kwargs) -> list:
        kwargs.update({
            'currencyPair': self.currency_pair(base_currency, quote_currency),
            'period': interval * 60
        })

        if since_last and 'ohlc' in self.last:
            kwargs.update({'start': self.last['ohlc']})

        result = self.api.returnChartData(**kwargs)

        if since_last:
            self.last['ohlc'] = result[-1]['date']

        def format_ohlc(datapoint):
            datapoint['datetime'] = pd.to_datetime(datapoint.pop('date'), unit='s')
            datapoint.pop('weightedAverage')
            datapoint.pop('quoteVolume')
            return datapoint

        return list(map(format_ohlc, result))

    def get_balance(self) -> dict:
        result = self.api.returnBalances()
        return result

    def limit_order(self, base_currency: str, buy_sell: str, price: float, volume: float,
                    quote_currency: str = 'USDT', wait: bool = True, **kwargs) -> dict:
        pair = self.currency_pair(base_currency, quote_currency)

        if buy_sell == 'buy':
            order_info = self.api.buy(price, volume, pair)
        else:
            assert buy_sell == 'sell'
            order_info = self.api.sell(price, volume, pair)

        # TODO: handle failed orders
        if not wait:
            return order_info

        self._wait_for_fill(order_info['orderNumber'], pair)
        trades = self.api.returnOrderTrades(order_info['orderNumber'])
        price = sum(trade['total'] for trade in trades)
        fee = sum(trade['fee'] for trade in trades)
        order_info.update({'trades': trades,
                           'price': price,
                           'fee': fee})
        return order_info

    def _wait_for_fill(self, order_id: int, pair: str) -> dict:
        filled = False
        while not filled:
            open_orders = self.api.returnOpenOrders(pair)
            target_orders = filter(lambda order: order['orderNumber'] == order_id, open_orders)
            filled = not target_orders
