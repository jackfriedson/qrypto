from typing import Any, Dict, List, Optional, Union

import pandas as pd


Timestamp = Union[int, pd.Timestamp]
"""A Timestamp can be either an integer representing a unix timestamp, or the
pandas.Timestamp class.
"""

OHLC = Dict[str, Union[Timestamp, float]]
"""
    {
        'datetime': ...,
        'open': ...,
        'high': ...,
        'low': ...,
        'close': ...,
        'volume': ...,
    }
"""

Order = Dict[str, Any]
"""
    {
        'id': ...,
        'status': ...,
        'buy_sell': ...,
        'ask_price': ...,
        'volume': ...,
        'fill_price': ...,
        'trades': ...,
        'fee': ...,
    }
"""

MaybeOrder = Optional[Order]
"""A MaybeOrder is either an Order or None."""

OrderBook = Dict[str, List[Dict[str, float]]]
"""
    {
        'asks': [ { 'price': ..., 'amount': ... }, ... ],
        'bids': [ { 'price': ..., 'amount': ... }, ... ]
    }
"""

Trade = Dict[str, float]
"""
        {
            'id': ... ,
            'timestamp': ...,
            'price': ...,
            'amount': ...,
        }
"""
