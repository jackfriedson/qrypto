import numpy as np

from cryptotrading.data.datasets import OHLCDataset


NON_NORMED_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'quoteVolume', 'avg']


class QLearnDataset(OHLCDataset):
    # actions = ['do_nothing', 'buy', 'sell']
    actions = ['buy', 'sell']

    def __init__(self, *args, precision: int = 10, fee: float = 0.002, **kwargs):
        self.precision = precision
        self.fee = fee
        self.open_price = None
        self.position = 'long'
        super(QLearnDataset, self).__init__(*args, **kwargs)

    @property
    def all(self):
        result = super(QLearnDataset, self).all
        result.drop(NON_NORMED_FIELDS, axis=1, inplace=True)
        return result

    @property
    def n_state_factors(self) -> int:
        return len(self.all.iloc[-1])

    @property
    def n_states(self) -> int:
        return 2 * (self.precision ** len(self.all.iloc[-1]))

    @property
    def n_actions(self):
        return len(self.actions)

    def state_vector(self, normalize=True):
        result = self.all.iloc[-1].values

        # Normalize values using existing data
        if normalize:
            result = result - self.all.mean()
            result = result / self.all.std()

        # result = np.append(result, self.cumulative_return)
        # result = np.append(result, 1. if self.open_price else -1.)
        result = result.values.reshape(1, len(result))
        return result

    @property
    def state(self):
        state = 0
        state_data = self.all.iloc[-1]

        for i, value in enumerate(state_data):
            discretized = value // self.precision
            if discretized == self.precision:
                discretized -= 1
            state += (self.precision**i) * discretized

        if self.open_price:
            state += self.n_states // 2

        if not np.isnan(state):
            state = int(state)

        return state

    def reset(self):
        self._data = None
        self._orders['buy'] = []
        self._orders['sell'] = []

    @property
    def period_return(self):
        return (self.close[-1] / self.close[-2]) - 1.

    @property
    def cumulative_return(self):
        if self.open_price:
            return (self.last / self.open_price) - 1.
        else:
            return 0.

    # def take_action(self, idx: int):
    #     self.add_order(self.actions[idx], {'price': self.last})

    #     if not self.open_price:
    #         if self.actions[idx] == 'buy':
    #             self.open_price = self.last
    #             return -self.fee
    #         else:
    #             return 0.
    #     else:
    #         if self.actions[idx] == 'sell':
    #             self.open_price = None
    #             return -self.fee
    #         else:
    #             return self.period_return

    def take_action_ls(self, idx: int):
        action = self.actions[idx]
        self.add_order(action, {'price': self.last})

        if self.position == 'long':
            if action == 'sell':
                self.position = 'short'
                return -self.fee
            else:
                return self.period_return
        else:
            if action == 'buy':
                self.position == 'long'
                return -self.fee
            else:
                return -self.period_return

    def test_action(self, idx: int, add_order: bool = True):
        if add_order:
            self.add_order(self.actions[idx], {'price': self.last})

        if self.actions[idx] == 'buy':
            if not self.open_price:
                self.open_price = self.last
            return 0.

        if self.actions[idx] == 'sell':
            if self.open_price:
                result = self.cumulative_return - (2 * self.fee)
                self.open_price = None
                return result
            return 0.

        return 0.
