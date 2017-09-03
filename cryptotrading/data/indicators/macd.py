from typing import List

import numpy as np

from cryptotrading.data.indicators import BaseIndicator


class MACD(BaseIndicator):
    indicator_name = 'macd'

    def _format_config(self, fast, slow, signal):
        return {
            'fastperiod': fast,
            'slowperiod': slow,
            'signalperiod': signal
        }

    def plot(self, axis):
        axis.plot(self.data.index, self.data['macdhist'])
        config_vals = ', '.join([str(v) for k, v in self.config.items()])
        axis.set_title(self.indicator_name.upper() + ' (' + config_vals + ')')

    @property
    def macd(self):
        return self.data['macdhist'].values

    def macd_slope(self, n: int = 3) -> float:
        if len(self.macd) < n:
            return np.nan

        x = np.arange(float(n))
        y = self.macd[-n:]
        line = np.polyfit(x, y, 1, full=True)
        return line[0][0]
