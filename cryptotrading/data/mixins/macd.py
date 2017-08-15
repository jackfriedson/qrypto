import numpy as np
import talib

class MACDMixin(object):

    def __init__(self, *args, **kwargs):
        self.fast, self.slow, self.signal = kwargs.pop('macd')
        super(MACDMixin, self).__init__(*args, **kwargs)

    def macd(self):
        return talib.MACD(self.close, fastperiod=self.fast, slowperiod=self.slow,
                          signalperiod=self.signal)

    def macd_slope(self, n=3):
        _, _, macdhist = self.macd()
        x = np.arange(float(n))
        y = macdhist[-n:]
        line = np.polyfit(x, y, 1, full=True)
        return line[0][0]
