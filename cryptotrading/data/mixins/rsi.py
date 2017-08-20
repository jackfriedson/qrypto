from typing import List

import talib
from pandas import Series


class RSIMixin(object):

    def __init__(self, *args, **kwargs):
        self.rsi_periods = kwargs.pop('rsi')
        super(RSIMixin, self).__init__(*args, **kwargs)

    def update(self, incoming_data: List[dict]) -> None:
        rsi_values = talib.RSI(self.close, timeperiod=self.rsi_periods)
        self._data['rsi'] = Series(rsi_values, index=self._data.index)
        try:
            super(RSIMixin, self).update(incoming_data)
        except AttributeError:
            pass

    @property
    def rsi(self):
        return self._data['rsi'].values
