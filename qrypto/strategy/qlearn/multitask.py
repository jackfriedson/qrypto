import numpy as np

from qrypto.models import RNNMultiTaskLearner
from qrypto.strategy import LearnStrategy


class MultitaskStrategy(LearnStrategy):
    tasks = ['direction', 'return']

    def __init__(self, *args, **kwargs):
        super(MultitaskStrategy, self).__init__(RNNMultiTaskLearner, 'rnn_multitask', *args, **kwargs)

    @staticmethod
    def _create_label(data):
        direction = 1 if data.period_return > 0 else 0
        # volatility = data.get_last('stddev')
        return np.array([direction, data.period_return])

    @staticmethod
    def _order_strategy(output, is_label: bool = False):
        return output[0] if is_label else np.argmax(output[0][0])
