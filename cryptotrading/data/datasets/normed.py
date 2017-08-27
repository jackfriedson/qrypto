from cryptotrading.data.datasets import OHLCDataset


NON_NORMED_FIELDS = ['open', 'high', 'low', 'close', 'volume']


class NormedDataset(OHLCDataset):
    """ Wraps an OHLC dataset to prevent access of non-normalized values.
    """

    @property
    def all(self):
        result = super(NormedDataset, self).all
        result.drop(NON_NORMED_FIELDS, axis=1, inplace=True)
        return result
