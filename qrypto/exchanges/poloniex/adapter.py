import time
from typing import Callable, List, Optional

import pandas as pd
from poloniex import PoloniexAPI

from qrypto.exchanges import BaseAPIAdapter, PrivateExchangeMixin, utils
from qrypto.types import MaybeOrder, OHLC, OrderBook, OrderInfo, Timestamp, Trade


class PoloniexAPIAdapter(BaseAPIAdapter, PrivateExchangeMixin):

    def __init__(self, key_path: str) -> None:
        apikey, secret = self.load_api_key(key_path)
        self.api = PoloniexAPI(self.apikey, secret)
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

    @staticmethod
    def _format_trades(trades):
        pass

    def get_order_book(self,
                       base_currency: str,
                       quote_currency: str = 'USDT') -> OrderBook:
        raise NotImplementedError('TODO')

    def get_balance(self) -> dict:
        result = self.api.returnBalances()
        return result

    def market_order(self):
        # TODO: Implement market orders using limit orders
        raise NotImplementedError('Poloniex does not support market orders')

    def limit_order(self,
                    base_currency: str,
                    buy_sell: str,
                    price: float,
                    volume: float,
                    quote_currency: str = DEFAULT_QUOTE_CURRENCY,
                    wait_for_fill: bool = False) -> MaybeOrder:
        pair = self.currency_pair(base_currency, quote_currency)
        order_fn = self.api.buy if buy_sell == 'buy' else self.api.sell
        order_info = order_fn(price, volume, pair)

        # TODO: handle failed order, return None
        order_info.update({'status': 'open',
                           'buy_sell': buy_sell,
                           'ask_price': price,
                           'volume': volume})

        filled_order_info = {}
        if wait_for_fill:
            self._wait_for_fill(order_info['orderNumber'], pair)
            trades = self.api.returnOrderTrades(order_info['orderNumber'])
            fill_price = sum(trade['total'] for trade in trades)
            fee = sum(trade['fee'] for trade in trades)
            filled_order_info = {
                'status': 'closed',
                'fill_price': fill_price,
                'trades': trades,
                'fee': fee
            }

        return self._format_order(order_info, filled_info=filled_order_info)

    @staticmethod
    def _format_order(order, filled_info):
        order_info = {
            'id': order['orderNumber'],
            'status': order['status'],
            'buy_sell': order['buy_sell'],
            'ask_price': order['ask_price'],
            'volume': order['volume']
        }
        order_info.update(filled_info)
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
