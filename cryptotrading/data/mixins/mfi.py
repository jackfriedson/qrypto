import talib


class MFIMixin(object):

    def __init__(self, *args, **kwargs):
        self.mfi_periods = kwargs.pop('mfi')
        super(MFIMixin, self).__init__(*args, **kwargs)

    def mfi(self):
        return talib.MFI(self.high[:-1], self.low[:-1], self.close[:-1], self.volume[:-1], timeperiod=self.mfi_periods)
