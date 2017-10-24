from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import Day


TONE_COL_IDX = 7


class GKGDataset(object):

    def __init__(self, freq, gkg_file: Path):
        freq = str(freq) + 'T'
        data_dicts = []
        with gkg_file.open() as f:
            headers = f.readline().split('\t')

            for line in f.readlines():
                row = line.split('\t')
                data = row[TONE_COL_IDX].split(',')
                data_dicts.append({
                    # Add a day to prevent data leakage
                    'date': pd.to_datetime(row[0]) + Day(),
                    'tone': float(data[0]),
                    'positive': float(data[1]),
                    'negative': float(data[2]),
                    'polarity': float(data[3])
                })

        all_data = pd.DataFrame(data_dicts)
        unique_dates = all_data.loc[:, 'date'].unique()

        rows = []
        for date in unique_dates:
            subset = all_data.loc[all_data['date'] == date]
            subset = subset.drop('date', axis=1)
            row = subset.agg('mean')
            row.name = date
            rows.append(row)

        self._data = pd.DataFrame(rows)
        self._data = self._data.resample(freq).pad()

    def between(self, start, end):
        return self._data.loc[start:end]
