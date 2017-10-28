import numpy as np

from qrypto.models import RNNRegressor
from qrypto.strategy import LearnStrategy


class RegressorStrategy(LearnStrategy):
    tasks = ['return']

    def __init__(self, *args, **kwargs):
        super(RegressorStrategy, self).__init__(RNNRegressor, 'rnn_regressor', *args, **kwargs)

    @staticmethod
    def _create_label(data):
        return np.array([data.period_return])

    @staticmethod
    def _order_strategy(output, is_label: bool = False):
        return 1 if output[0] > 0 else 0
