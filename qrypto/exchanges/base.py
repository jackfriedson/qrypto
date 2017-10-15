from abc import ABC, abstractmethod, abstractstaticmethod
from typing import List, Optional

import pandas as pd

from qrypto.types import MaybeOrder, OHLC, OrderBook, Timestamp, Trade


DEFAULT_QUOTE_CURRENCY = 'USD'


class BaseAPIAdapter(ABC):
    """Abstract base class for creating a wrapper around an exchange's API.

    Note: Implementing classes may choose to return more data than described here in
    their responses (e.g. average price, quote volume, etc.), but all must return
    AT LEAST the data described here.
    """

    @abstractstaticmethod
    def currency_pair(base_currency: str, quote_curency: str) -> str:
        """Combines the base and quote currencies into a currency pair that will
        be recognized by the exchange.

        :param base_currency:
        :param quote_currency:
        :returns: a currency pair recognizable by the exchange
        """
        pass

    @abstractmethod
    def get_ohlc(self,
                 base_currency: str,
                 quote_currency: str = DEFAULT_QUOTE_CURRENCY,
                 interval: int = 1,
                 start: Optional[Timestamp] = None,
                 end: Optional[Timestamp] = None) -> List[OHLC]:
        """Gets OHLC (candlestick) data from the exchange.

        :param base_currency:
        :param quote_currency:
        :param interval: the interval (in minutes) of the data to be fetched
        :param start: a timestamp representing the beginning of the requested data
        :param end: a timestamp representing the end of the requested data
        :returns: list of OHLC time periods
        """
        pass

    @abstractmethod
    def get_trades(self,
                   base_currency: str,
                   quote_currency: str = DEFAULT_QUOTE_CURRENCY,
                   since: Optional[Timestamp] = None) -> List[Trade]:
        """Gets the most recent trades.

        :param base_currency:
        :param quote_currency:
        :param since: only return trades occuring after this timestamp
        :returns: list of the most recent trades

        """
        pass

    @abstractmethod
    def get_order_book(self,
                       base_currency: str,
                       quote_currency: str = DEFAULT_QUOTE_CURRENCY) -> OrderBook:
        """Gets the current order book.

        :param base_currency:
        :param quote_currency:
        :returns: the current order book
        """
        pass


class PrivateExchangeMixin(ABC):
    """Abstract base class to provide additional functionality for private exchanges.
    Includes methods for placing, cancelling, and managing orders, as well as
    other account-related actions.
    """

    @abstractmethod
    def market_order(self,
                     base_currency: str,
                     buy_sell: str,
                     volume: float,
                     quote_currency: str = DEFAULT_QUOTE_CURRENCY) -> MaybeOrder:
        """Places a market order.

        :param base_currency:
        :param buy_sell: whether or not the order is a 'buy' or a 'sell'
        :param volume: how many units of the base currency to buy/sell
        :param quote_currency:
        :returns: info about the order that was placed if successful, None otherwise
        """
        pass

    @abstractmethod
    def limit_order(self,
                    base_currency: str,
                    buy_sell: str,
                    price: float,
                    volume: float,
                    quote_currency: str = DEFAULT_QUOTE_CURRENCY,
                    wait_for_fill: bool = False) -> MaybeOrder:
        """Places a limit order at the specified price.

        :param base_currency:
        :param buy_sell: whether or not the order is a 'buy' or a 'sell'
        :param price: the asking price of the order
        :param volume: how many units of the base currency to buy/sell
        :param quote_currency:
        :param wait_for_fill: if True, wait until the order is filled and return more complete info
                              (e.g. will include fill_price)
        :returns: info about the order that was placed if successful, None otherwise
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancels the specified order.

        :param order_id: the ID of the order to cancel
        :returns: True if the order was successfully cancelled, False otherwise
        """
        pass
