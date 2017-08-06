"""
"""
import time

class BaseStrategy(object):

    def __init__(self, base_currency, exchange, unit, quote_currency='USD', sleep_duration=(60, 120)):
        """

        """
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.exchange = exchange
        self.unit = unit
        self.positions = []

        if isinstance(sleep_duration, int):
            self.no_position_sleep_duration = self.active_position_sleep_duration = sleep_duration
        else:
            self.no_position_sleep_duration, self.active_position_sleep_duration = sleep_duration

    def run(self):
        while True:
            self.update()

            # Ensure we only ever have one open position at a time
            # Note: a position may contain multiple orders
            if not self.positions:
                self.open_condition()
                time.sleep(self.no_position_sleep_duration)
            else:
                self.close_condition()
                time.sleep(self.active_position_sleep_duration)

    def update(self):
        raise NotImplementedError

    def open_condition(self):
        raise NotImplementedError

    def close_condition(self):
        raise NotImplementedError

    def open_position(self):
        raise NotImplementedError

    def cancel_positions(self):
        if self.positions:
            for txid in self.positions:
                self.exchange.cancel_order(txid)

    def cancel_all_if_any_close(self):
        orders = self.exchange.get_orders_info(self.positions)
        for txid, order_info in orders.items():
            status = order_info['status']
            print('Order {} is {}'.format(txid, status))
            if status in ['closed', 'canceled', 'expired']:
                self.positions.remove(txid)
                self.cancel_positions()
