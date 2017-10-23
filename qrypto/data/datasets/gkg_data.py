from pathlib import Path

import pandas as pd


TONE_COL_IDX = 7


class GKGDataset(object):

    def __init__(self, gkg_file: Path):
        data_dicts = []
        with gkg_file.open() as f:
            headers = f.readline().split('\t')

            for line in f.readlines():
                row = line.split('\t')
                data = row[TONE_COL_IDX].split(',')
                data_dicts.append({
                    'date': pd.to_datetime(row[0]),
                    'tone': float(data[0]),
                    'positive': float(data[1]),
                    'negative': float(data[2]),
                    'polarity': float(data[3]),
                    'activity_ref_density': float(data[4]),
                    'self_group_ref_density': float(data[5])
                })

        self._data = pd.DataFrame(data_dicts)
        # import ipdb; ipdb.set_trace()
