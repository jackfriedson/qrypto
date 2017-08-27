from typing import List

from talib.abstract import RSI


class RSIMixin(object):

    def __init__(self, *args, **kwargs):
        self.rsi_periods = kwargs.pop('rsi')
        super(RSIMixin, self).__init__(*args, **kwargs)

    def update(self, incoming_data: List[dict]) -> None:
        self._indicators['rsi'] = RSI(self._data, timeperiod=self.rsi_periods).to_frame('rsi')

        try:
            super(RSIMixin, self).update(incoming_data)
        except AttributeError:
            pass

    @property
    def rsi(self):
        return self._indicators['rsi'].values
