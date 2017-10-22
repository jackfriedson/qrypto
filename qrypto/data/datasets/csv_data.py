from typing import Callable, Optional, List, Union
from pathlib import Path

import pandas as pd


class CSVDataset(object):

    def __init__(self, frequency: int, csv_configs: List[dict]):
        """
        {
            'path': ...,
            'name': ...,
            'headers': ...,
            'date_converter': ...
        }
        """
        self._data = None
        self._freq = str(frequency) + 'T'

        for csv_info in csv_configs:
            self.add_column_from_csv(**csv_info)

    def add_column_from_csv(self,
                            path: Path,
                            name: Union[str, List[str]],
                            headers: bool = False,
                            date_converter: Optional[Callable] = None):
        if date_converter is None:
            date_converter = lambda x: pd.to_datetime(x)

        headers = 0 if headers else None
        col_names = ['datetime']
        if isinstance(name, str):
            name = [name]
        col_names.extend(name)

        csv_df = pd.read_csv(path, header=headers, index_col=0, names=col_names, converters={0: date_converter})
        csv_df = csv_df.resample(self._freq).pad()
        self._add_dataframe(csv_df)

    def _add_dataframe(self, new_df: pd.DataFrame):
        if self._data is None:
            self._data = new_df
        else:
            self._data = self._data.join(new_df, how='outer')

    def between(self, start, end):
        return self._data.loc[start:end]

    @property
    def all(self):
        return self._data