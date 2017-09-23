from typing import List

from qrypto.data.indicators import BasicIndicator


class DifferenceIndicator(BasicIndicator):

    def __init__(self, indicator_name, config: dict = None):
        super(DifferenceIndicator, self).__init__(indicator_name, config=config)

    def update(self, core_data: List[dict]) -> None:
        result = self.fn(core_data, **self.config)
        self.data = (result.iloc[:, 0] - result.iloc[:, 1]).to_frame(self.indicator_name + '_diff')
