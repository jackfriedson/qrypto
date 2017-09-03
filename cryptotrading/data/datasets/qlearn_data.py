import numpy as np

from cryptotrading.data.datasets import OHLCDataset


NON_NORMED_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'quoteVolume', 'avg']


class QLearnDataset(OHLCDataset):

    def __init__(self, *args, precision: int = 10, fee: float = 0.001, **kwargs):
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
        return 2 * (self.precision ** len(self.all.iloc[-1]))

    @property
    def state(self):
        state = 0
        state_data = self.all.iloc[-1]

        for i, value in enumerate(state_data):
            discretized = value // self.precision
            if discretized == self.precision:
                discretized -= 1
            state += (self.precision**i) * discretized

        if self.open_position:
            state += self.n_states // 2

        if not np.isnan(state):
            state = int(state)

        return state

    def reset(self):
        self._data = None

    @property
    def period_return(self):
        return (self.close[-1] / self.close[-2]) - 1.

    @property
    def cumulative_return(self):
        return (self.last / self.open_position) - 1.

    def take_action(self, action: str):
        if action == 'do_nothing':
            if not self.open_position:
                return self.state, 0.
            else:
                return self.state, self.period_return

        if action == 'buy':
            self.add_order('buy', {'price': self.last})
            if not self.open_position:
                self.open_position = self.last
                return self.state, -self.fee
            else:
                return self.state, 0.

        if action == 'sell':
            self.add_order('sell', {'price': self.last})
            if self.open_position:
                # TODO: Discount unrealized gains/losses to incentivize selling at a high
                self.open_position = None
                return self.state, -self.fee
            else:
                return self.state, 0.

    def test_action(self, action: str):
        if action == 'buy':
            self.add_order('buy', {'price': self.last})
            if not self.open_position:
                self.open_position = self.last
            return 0.

        if action == 'sell':
            self.add_order('sell', {'price': self.last})
            if self.open_position:
                result = self.cumulative_return - (2 * self.fee)
                self.open_position = None
                return result
            return 0.

        return 0.
