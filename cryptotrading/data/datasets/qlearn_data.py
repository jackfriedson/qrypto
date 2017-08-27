from cryptotrading.data.datasets import OHLCDataset


NON_NORMED_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'quoteVolume', 'avg']


class QLearnDataset(OHLCDataset):

    @property
    def all(self):
        result = super(QLearnDataset, self).all
        result.drop(NON_NORMED_FIELDS, axis=1, inplace=True)
        return result

    @property
    def state(self, precision=10) -> int:
        state = 0
        state_data = self.all.iloc[-1]

        for i, value in enumerate(state_data):
            state += (precision**i) * (value // precision)

        return state
