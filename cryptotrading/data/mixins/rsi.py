import talib

class RSIMixin(object):

    def __init__(self, *args, **kwargs):
        self.rsi_periods = kwargs.pop('rsi')
        super(RSIMixin, self).__init__(*args, **kwargs)

    def rsi(self):
        return talib.RSI(self.close, timeperiod=self.rsi_periods)
