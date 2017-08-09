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
        for txid in self.positions:
            self.exchange.cancel_order(txid)
            self.positions.remove(txid)

    def any_orders_closed(self):
        """ Checks if any open positions have been closed.

        Returns true if any open positions are closed, canceled, or expired, and removes
        those positions from the list.
        """
        orders = self.exchange.get_orders_info(self.positions)
        for txid, order_info in orders.items():
            status = order_info['status']
            log.info('Order %s is %s', txid, status)

            if status in ['closed', 'canceled', 'expired']:
                log.info('Order %s closed at %s',txid, order_info['cost'],
                         extra={
                            'event_name': 'order_' + status,
                            'event_data': order_info
                         })
                self.positions.remove(txid)
                return True
        return False

    def wait_for_order_close(self, txid, sleep_inverval=2):
        order_open = True
        while order_open:
            time.sleep(sleep_inverval)
            order_info = self.exchange.get_orders_info([txid]).get(txid)
            order_open = order_info['status'] not in ['closed', 'canceled', 'expired']
            if order_open:
                log.info('Order %s still open', txid)

        log.info('Order %s closed @ %s', txid, order_info['cost'],
            extra={
                'event_name': 'order_closed',
                'event_data': order_info
            })
        return order_info
