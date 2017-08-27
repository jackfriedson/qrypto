from typing import List

from cryptotrading.data.datasets import OHLCDataset
from cryptotrading.data.mixins import MACDMixin, MFIMixin, RSIMixin


class QLearnDataset(object):
    """ Wraps an OHLC dataset to prevent access of non-normalized values.
    """

    class _Dataset(OHLCDataset, MACDMixin, MFIMixin, RSIMixin):
        pass

    def __init__(self, *args, **kwargs):
        self._data_internal = self._Dataset(*args, **kwargs)

    def update(self, incoming_data: List[dict]) -> None:
        self._data_internal.update(incoming_data)