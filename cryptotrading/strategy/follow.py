from cryptotrading.data.datasets import OHLCDataset
from cryptotrading.strategy import BaseStrategy


class _Dataset(OHLCDataset):
    pass


class FollowStrategy(BaseStrategy):

    def __init__(self, base_currency, exchange, unit, quote_currency='USD', follow_currency='BTC',
                 sleep_duration='1'):
        super(FollowStrategy, self).__init__(base_currency, exchange, unit, quote_currency=quote_currency,
                                             sleep_duration=sleep_duration)
        self.follow_currency = follow_currency

        self.data = _Dataset()

    def update(self):
        new_data = self.exchange.recent_ohlc(self.base_currency, self.quote_currency)
        self.data.add_all(new_data)

    def check_condition(self):
        pass

    def open_position(self):
        pass
