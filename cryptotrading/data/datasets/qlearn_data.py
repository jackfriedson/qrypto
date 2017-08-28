from cryptotrading.data.datasets import OHLCDataset


NON_NORMED_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'quoteVolume', 'avg']


class QLearnDataset(OHLCDataset):

    def __init__(self, *args, precision: int = 10, fee: float = 0.002, **kwargs):
        self.precision = precision
        self.fee = fee
        self.open_position = None
        super(QLearnDataset, self).__init__(*args, **kwargs)

    @property
    def all(self):
        result = super(QLearnDataset, self).all
        result.drop(NON_NORMED_FIELDS, axis=1, inplace=True)
        return result

    @property
    def n_states(self) -> int:
        return 2 * len(self.all.iloc[-1]) * self.precision

    @property
    def state(self) -> int:
        state = 0
        state_data = self.all.iloc[-1]

        for i, value in enumerate(state_data):
            state += (self.precision**i) * (value // self.precision)

        if self.open_position:
            state += self.n_states // 2

        return state

    def take_action(self, action: str) -> float:
        if action == 'buy':
            self.open_position = self.last
            return int(self.state), -self.fee
        elif action == 'sell':
            # TODO: Discount unrealized gains/losses to incentivize selling at a high
            self.open_position = None
            return int(self.state), -self.fee
        elif self.open_position:
            return int(self.state), (self.close[-1] / self.close[-2]) - 1.
        else:
            return int(self.state), 0.
