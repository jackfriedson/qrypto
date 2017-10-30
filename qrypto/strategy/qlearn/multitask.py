import numpy as np

from qrypto.models import RNNMultiTaskLearner
from qrypto.strategy import LearnStrategy


class MultitaskStrategy(LearnStrategy):
    tasks = ['volatility', 'return']

    def __init__(self, *args, **kwargs):
        super(MultitaskStrategy, self).__init__(RNNMultiTaskLearner, 'rnn_multitask', *args, **kwargs)

    @staticmethod
    def _create_label(data):
        volatility = data.get_last('stddev')
        return np.array([volatility, data.period_return])

    @staticmethod
    def _order_strategy(output, is_label: bool = False):
        return 1 if output[1] > 0 else 0
