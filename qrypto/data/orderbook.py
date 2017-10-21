import csv
from pathlib import Path
from typing import List, Union

from orderbook import OrderBook
import pandas as pd


class OrderBook(object):

    def __init__(self, trades_csv: Union[Path, List[Path]], ohlc_data: pd.DataFrame):
        """
        :param trades_csv: CSV file (or files) of trade data, where first column is
                           unix timestamp of trade, second column is limit price,
                           and third column is volume
        """
        columns = ['datetime', 'price', 'volume']
        converters = {
            'datetime': lambda ts: pd.to_datetime(ts, unit='s')
        }
        # TODO: support multiple CSV files
        self._all_trades = pd.read_csv(trades_csv, header=None, names=columns,
                                       converters=converters)
        self._order_book = OrderBook()

