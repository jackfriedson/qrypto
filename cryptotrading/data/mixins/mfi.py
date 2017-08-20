from typing import List

import talib
from pandas import Series


class MFIMixin(object):

    def __init__(self, *args, **kwargs):
        self.mfi_periods = kwargs.pop('mfi')
        super(MFIMixin, self).__init__(*args, **kwargs)

    def update(self, incoming_data: List[dict]) -> None:
        mfi_values = talib.MFI(self.high, self.low, self.close, self.volume, timeperiod=self.mfi_periods)
        self._data['mfi'] = Series(mfi_values, index=self._data.index)
        try:
            super(MFIMixin, self).update(incoming_data)
        except AttributeError:
            pass

    @property
    def mfi(self):
        return self._data['mfi'].values
