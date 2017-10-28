import numpy as np

from qrypto.models import RNNClassifier
from qrypto.strategy import LearnStrategy


class ClassifierStrategy(LearnStrategy):
    tasks = ['direction']

    def __init__(self, *args, **kwargs):
        super(ClassifierStrategy, self).__init__(RNNClassifier, 'rnn_classifier', *args, **kwargs)

    @staticmethod
    def _create_label(data):
        label = 1 if data.period_return > 0 else 0
        return np.array([label])

    @staticmethod
    def _order_strategy(output, is_label: bool = False):
        return output[0] if is_label else np.argmax(output[0])
