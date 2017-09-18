from typing import List

from pandas import Series
from talib.abstract import Function


class BaseIndicator(object):
    indicator_name = None

    def __init__(self, *args, config_dict: dict = None):
        if not self.indicator_name:
            raise NotImplementedError('Subclasses of BaseIndicator must specify an indicator name')

        self.fn = Function(self.indicator_name)
        self.data = None

        if config_dict is not None:
            self.config = config_dict
        else:
            self.config = self._format_config(*args)

    def __getattr__(self, name):
        if name == self.indicator_name:
            return self.data[self.indicator_name].values
        raise AttributeError

    def _format_config(self, *args):
        raise NotImplementedError

    def update(self, core_data: List[dict]) -> None:
        data = self.fn(core_data, **self.config)
        if isinstance(data, Series):
            data = data.to_frame(self.indicator_name)
        self.data = data

    def plot(self, axis):
        axis.plot(self.data.index, self.data)
        config_vals = ', '.join([str(v) for k, v in self.config.items()])
        axis.set_title(self.indicator_name.upper() + ' (' + config_vals + ')')


class BasicIndicator(BaseIndicator):
    def __init__(self, name: str, config: dict = None) -> None:
        self.indicator_name = name
        super(BasicIndicator, self).__init__(config_dict=config or {})

    @property
    def suffix(self):
        return '_' + ','.join([str(v) for k, v in self.config.items()])
