

from cryptotrading.strategy.core import BaseStrategy

class FollowStrategy(BaseStrategy):

    def __init__(self, base_currency, exchange, unit, quote_currency='USD', follow_currency='BTC',
                 sleep_duration='1'):
        super(FollowStrategy, self).__init__(base_currency, exchange, unit, quote_currency=quote_currency,
                                             sleep_duration=sleep_duration)
        self.follow_currency = follow_currency

    def update(self):
        pass

    def check_condition(self):
        pass

    def open_position(self):
        pass
