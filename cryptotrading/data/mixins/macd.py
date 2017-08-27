from typing import List

import numpy as np
from talib.abstract import MACD


class MACDMixin(object):

    def __init__(self, *args, **kwargs):
        self.fast, self.slow, self.signal = kwargs.pop('macd')
        super(MACDMixin, self).__init__(*args, **kwargs)

    def update(self, incoming_data: List[dict]) -> None:
        self._indicators['macd'] = MACD(self._data, fastperiod=self.fast, slowperiod=self.slow,
                                        signalperiod=self.signal)

        try:
            super(MACDMixin, self).update(incoming_data)
        except AttributeError:
            pass

    @property
    def macd(self):
        return self._indicators['macd']['macdhist'].values

    def macd_slope(self, n: int = 3) -> float:
        if len(self.macd) < n:
            return np.nan

        x = np.arange(float(n))
        y = self.macd[-n:]
        line = np.polyfit(x, y, 1, full=True)
        return line[0][0]
