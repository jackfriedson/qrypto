import numpy as np

from qrypto.models import RegressorVarianceModel
from qrypto.strategy import LearnStrategy


class RegVarianceStrategy(LearnStrategy):
    tasks = ['return', 'variance']

    def __init__(self, *args, **kwargs):
        super(RegVarianceStrategy, self).__init__(RegressorVarianceModel, 'rnn_multitask', *args, **kwargs)

    @staticmethod
    def _create_label(data):
        return np.array([data.period_return])

    @staticmethod
    def _order_strategy(output, is_label: bool = False):
        if is_label:
            return 1 if output[0] > 0 else 0
        else:
            return 1 if output[0][0] > 0 else 0

        # pred_return = output[0][0]
        # pred_variance = output[1][0]
        # if pred_variance > .5 or pred_return < 0:
        #     return 0
        # return 1