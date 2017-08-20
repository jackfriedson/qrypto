"""
"""
import logging
import time
from typing import Tuple


log = logging.getLogger(__name__)


class BaseStrategy(object):

    def __init__(self, base_currency: str, exchange, unit: float, quote_currency: str = 'USD',
                 sleep_duration: Tuple[int, int] = (60, 120)):
        """
        """
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.exchange = exchange
        self.unit = unit
        self.position = None
        self.indicators = {}

        if isinstance(sleep_duration, int):
            self.passive_sleep_duration = self.active_sleep_duration = sleep_duration
        else:
            self.passive_sleep_duration, self.active_sleep_duration = sleep_duration

    def run(self) -> None:
        while True:
            self.update()

            if not self.position:
                if self.should_open():
                    self.open_position()
                else:
                    time.sleep(self.passive_sleep_duration)
            else:
                if self.should_close():
                    self.close_position()
                else:
                    time.sleep(self.active_sleep_duration)

    def update(self) -> None:
        raise NotImplementedError

    def should_open(self) -> bool:
        raise NotImplementedError

    def should_close(self) -> bool:
        raise NotImplementedError

    def open_position(self) -> None:
        raise NotImplementedError

    def close_position(self) -> None:
        raise NotImplementedError

    def cancel_all(self) -> None:
        for txid in self.position['orders']:
            self.exchange.cancel_order(txid)
            self.position['orders'].remove(txid)

    def any_orders_closed(self) -> bool:
        """ Checks if any open positions have been closed.

        Returns true if any open positions are closed, canceled, or expired, and removes
        those positions from the list.
        """
        orders = self.exchange.get_orders_info(self.position['orders'])
        for txid, order_info in orders.items():
            if order_info['status'] == 'closed':
                log.info('Order %s closed at %s', txid, order_info['cost'])
                self.position['orders'].remove(txid)
                return True
        return False

    def wait_for_order_close(self, txid: str, sleep_inverval: int = 2) -> dict:
        order_open = True
        while order_open:
            time.sleep(sleep_inverval)
            order_info = self.exchange.get_order_info(txid)
            order_open = order_info['status'] not in ['closed', 'canceled', 'expired']
        return order_info
