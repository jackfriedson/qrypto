import numpy as np

from cryptotrading.data.datasets import OHLCDataset


NON_NORMED_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'quoteVolume', 'avg']


class QLearnDataset(OHLCDataset):

    def __init__(self, *args, precision: int = 10, fee: float = 0., **kwargs):
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

    def take_action(self, action: str):
        if action == 'do_nothing':
            if not self.open_position:
                return self.state, 0.
            else:
                period_return = (self.close[-1] / self.close[-2]) - 1.
                return self.state, period_return

        if action == 'buy_sell':
            if not self.open_position:
                self.open_position = self.last
                self.add_order('buy', {'price': self.last})
                return self.state, -self.fee
            else:
                # TODO: Discount unrealized gains/losses to incentivize selling at a high
                profit_loss = (self.last / self.open_position) - 1.
                self.open_position = None
                self.add_order('sell', {'price': self.last})
                return self.state, -self.fee

    def test_action(self, action: str):
        if action == 'do_nothing':
            return 0.

        if not self.open_position:
            self.open_position = self.last
            self.add_order('buy', {'price': self.last})
            return 0.

        if self.open_position:
            profit_loss = (self.last / self.open_position) - 1.
            self.open_position = None
            self.add_order('sell', {'price': self.last})
            return profit_loss - (2 * self.fee)
