import csv
from pathlib import Path

import pandas as pd


class OrderBook(object):

    def __init__(self, trades_csv: Path, ohlc_data: pd.Dataframe):
        columns = ['datetime', 'price', 'volume']
        converters = {
            'datetime': lambda ts: pd.to_datetime(ts, unit='s')
        }
        self._all_trades = pd.read_csv(trades_csv, header=None, names=columns,
                                       index_col=0, converters=converters)



