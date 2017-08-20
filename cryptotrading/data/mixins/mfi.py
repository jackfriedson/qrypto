import talib


class MFIMixin(object):

    def __init__(self, *args, **kwargs):
        self.mfi_periods = kwargs.pop('mfi')
        super(MFIMixin, self).__init__(*args, **kwargs)

    def mfi(self):
        return talib.MFI(self.high, self.low, self.close, self.volume, timeperiod=self.mfi_periods)
