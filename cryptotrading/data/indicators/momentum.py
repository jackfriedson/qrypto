from typing import List
from talib.abstract import MOM


class MomentumMixin(object):

    def __init__(self, *args, **kwargs):
        self.window = kwargs.pop('momentum')
        super(MomentumMixin, self).__init__(*args, **kwargs)

    def update(self, incoming_data: List[dict]) -> None:
        self._indicators['momentum'] = MOM(self._data, timeperiod=self.window).to_frame('momentum')

        try:
            super(MomentumMixin, self).update(incoming_data)
        except AttributeError:
            pass

    @property
    def momentum(self):
        return self._indicators['momentum'].values
