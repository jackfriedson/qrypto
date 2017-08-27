from typing import List

import numpy as np
import talib
from pandas import Series


class MACDMixin(object):

    def __init__(self, *args, **kwargs):
        self.fast, self.slow, self.signal = kwargs.pop('macd')
        super(MACDMixin, self).__init__(*args, **kwargs)

    def update(self, incoming_data: List[dict]) -> None:
        _, _, macd_values = talib.MACD(self.close, fastperiod=self.fast, slowperiod=self.slow,
                                       signalperiod=self.signal)
        self._data['macd'] = Series(macd_values, index=self._data.index)
        try:
            super(MACDMixin, self).update(incoming_data)
        except AttributeError:
            pass

    @property
    def macd(self):
        return self._data['macd'].values

    def macd_slope(self, n: int = 3) -> float:
        if len(self.macd) < n:
            return np.nan

        x = np.arange(float(n))
        y = self.macd[-n:]
        line = np.polyfit(x, y, 1, full=True)
        return line[0][0]
