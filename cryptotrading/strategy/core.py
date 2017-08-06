"""
"""

class BaseStrategy(object):

    def __init__(self, base_currency, exchange, unit, quote_currency='USD', sleep_duration=1):
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.exchange = exchange
        self.unit = unit
        self.sleep_duration = sleep_duration
        self.position = []

    def run(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def check_condition(self):
        raise NotImplementedError

    def open_position(self):
        raise NotImplementedError

    def close_position(self):
        raise NotImplementedError

    def cancel_position(self):
        if self.position:
            for txid in self.position:
                self.exchange.cancel_order(txid)
