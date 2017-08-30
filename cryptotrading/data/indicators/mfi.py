from typing import List

from talib.abstract import MFI


class MFIMixin(object):

    def __init__(self, *args, **kwargs):
        self.mfi_periods = kwargs.pop('mfi')
        super(MFIMixin, self).__init__(*args, **kwargs)

    def update(self, incoming_data: List[dict]) -> None:
        self._indicators['mfi'] = MFI(self._data, timeperiod=self.mfi_periods).to_frame('mfi')

        try:
            super(MFIMixin, self).update(incoming_data)
        except AttributeError:
            pass

    @property
    def mfi(self):
        return self._indicators['mfi'].values
