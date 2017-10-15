import time
from typing import Callable, List, Optional

import pandas as pd
from poloniex import PoloniexAPI

from qrypto.exchanges import BaseAPIAdapter, OHLC, OrderBook, Timestamp, Trade, utils


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

    def get_ohlc(self,
                 base_currency: str,
                 quote_currency: str = 'USDT',
                 interval: int = 5,
                 start: Optional[Timestamp] = None,
                 end: Optional[Timestamp] = None) -> List[OHLC]:
        kwargs = {
            'currencyPair': self.currency_pair(base_currency, quote_currency),
            'period': interval * 60,
        }

        if start is not None:
            kwargs.update({'start': utils.to_unixtime(start)})
        elif 'ohlc' in self.last:
            kwargs.update({'start': self.last['ohlc']})

        if end is not None:
            kwargs.update({'end': utils.to_unixtime(end)})

        result = self.api.returnChartData(**kwargs)

        # Record the most recent timestamp we've seen
        self.last['ohlc'] = result[-1]['date']

        return self._format_ohlc(result)

    def get_trades(self,
                   base_currency: str,
                   quote_currency: str = 'USDT',
                   since: Optional[Timestamp] = None) -> List[Trade]:
        raise NotImplementedError('TODO')

    def get_order_book(self,
                       base_currency: str,
                       quote_currency: str = 'USDT') -> OrderBook:
        raise NotImplementedError('TODO')

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

    @staticmethod
    def _format_ohlc(data):
        return [{
            'datetime': pd.to_datetime(d['date'], unit='s'),
            'open': d['open'],
            'high': d['high'],
            'low': d['low'],
            'close': d['close'],
            'volume': d['volume']
        } for d in data]
