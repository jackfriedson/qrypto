from typing import List, Optional

import cryptowatch as cw
import pandas as pd

from qrypto.exchanges import BaseAPIAdapter
from qrypto.types import OHLC, OrderBook, Timestamp, Trade


class CryptowatchAPIAdapter(BaseAPIAdapter):

    def __init__(self, exchange: str = 'kraken') -> None:
        self.exchange = exchange

    @staticmethod
    def currency_pair(base_currency: str, quote_currency: str) -> str:
        result = base_currency + quote_currency
        return result.lower()

    def get_ohlc(self,
                 base_currency: str,
                 quote_currency: str,
                 interval: int = 1,
                 start: Optional[Timestamp] = None,
                 end: Optional[Timestamp] = None) -> List[OHLC]:
        client = cw.Client.MarketClient(self.exchange,
                                        self.currency_pair(base_currency, quote_currency))
        period = interval * 60

        kwargs = {
            # TODO: consider adding support for multiple periods in a single call
            'periods': [str(period)]
        }

        if start:
            if isinstance(start, pd.Timestamp):
                start = start.astype(int)
            kwargs['after'] = start

        if end:
            if isinstance(end, pd.Timestamp):
                end = end.astype(int)
            kwargs['before'] = end

        result = client.GetOHLC(**kwargs)[period]
        return self._format_ohlc(result)

    def get_trades(self,
                   base_currency: str,
                   quote_currency: str,
                   since: Optional[Timestamp] = None) -> List[Trade]:
        raise NotImplementedError('TODO')

    def get_order_book(self, base_currency: str, quote_currency: str) -> OrderBook:
        raise NotImplementedError('TODO')

    @staticmethod
    def _format_ohlc(candles: List[cw.Msg.Candle]) -> List[OHLC]:
        return [{
            'datetime': pd.to_datetime(c.open_time, unit='s'),
            'open': c.open,
            'high': c.high,
            'low': c.low,
            'close': c.close,
            'volume': c.volume
        } for c in candles]

