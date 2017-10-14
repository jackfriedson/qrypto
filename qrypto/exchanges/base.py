from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Dict, List, Optional, Union

import pandas as pd


DEFAULT_QUOTE_CURRENCY = 'USD'

Timestamp = Union[int, pd.Timestamp]
"""A Timestamp can be either a unix timestamp, or the builtin pandas Timestamp."""

OHLC = Dict[str, Union[Timestamp, float]]
"""An OHLC is a dictionary of the following form:
    {
        'datetime': ,
        'open': ...,
        'high': ...,
        'low': ...,
        'close': ...,
        'volume': ...,
    }
"""

Trade = Dict[str, float]
"""Trades are dictionaries of the following form:
        {
            'id': ... ,
            'timestamp': ...,
            'price': ...,
            'amount': ...,
        }
"""

OrderBook = Dict[str, List[Dict[str, float]]]
"""An OrderBook is a dictionary of the following form:
    {
        'asks': [ { 'price': ..., 'amount': ... }, ... ],
        'bids': [ { 'price': ..., 'amount': ... }, ... ]
    }
"""


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

