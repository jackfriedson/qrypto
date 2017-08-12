"""
"""
import logging
import time


log = logging.getLogger(__name__)


class BaseStrategy(object):

    def __init__(self, base_currency, exchange, unit, quote_currency='USD', sleep_duration=(60, 120)):
        """
        """
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.exchange = exchange
        self.unit = unit
        self.positions = []
        self.indicators = {}

        if isinstance(sleep_duration, int):
            self.passive_sleep_duration = self.active_sleep_duration = sleep_duration
        else:
            self.passive_sleep_duration, self.active_sleep_duration = sleep_duration

    def run(self):
        while True:
            self.update()

            if not self.positions:
                if self.should_open():
                    self.open_position()
                else:
                    time.sleep(self.passive_sleep_duration)
            else:
                if self.should_close():
                    self.close_position()
                else:
                    time.sleep(self.active_sleep_duration)

    def update(self):
        raise NotImplementedError

    def should_open(self):
        raise NotImplementedError

    def should_close(self):
        raise NotImplementedError

    def open_position(self):
        raise NotImplementedError

    def close_position(self):
        raise NotImplementedError

    def cancel_all(self):
        log.info('Cancelling remaining orders...')
        for txid in self.positions:
            self.exchange.cancel_order(txid)
            self.positions.remove(txid)
