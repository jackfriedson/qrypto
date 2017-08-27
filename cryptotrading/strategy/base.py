"""
"""
import logging
import time
from typing import Tuple

from cryptotrading.data.datasets import OHLCDataset


log = logging.getLogger(__name__)


class BaseStrategy(object):

    class _Dataset(OHLCDataset):
        pass

    def __init__(self, base_currency: str, exchange, unit: float, quote_currency: str = 'USD',
                 sleep_duration: Tuple[int, int] = (60, 120), **kwargs):
        """
        """
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.exchange = exchange
        self.unit = unit
        self.position = None
        self.data = self._Dataset(**kwargs)

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
        log.info('Opening position...')
        # TODO: use last ask instead of 0.1%
        limit_price = self.data.last * 1.001
        order_info = self.exchange.limit_order(self.base_currency, 'buy', limit_price, self.unit,
                                               quote_currency=self.quote_currency)
        open_price = order_info['price']
        self.position = {'open': open_price}
        log.info('Position opened @ %.2f; Fee: %.2f', open_price, order_info['fee'])

    def close_position(self) -> None:
        log.info('Closing position...')
        # TODO: use last bid instead of 0.1%
        limit_price = self.data.last * 0.999
        order_info = self.exchange.limit_order(self.base_currency, 'sell', limit_price, self.unit,
                                               quote_currency=self.quote_currency)
        close_price = order_info['price']
        profit_loss = 100. * ((close_price / self.position['open']) - 1.)
        self.position = None
        log.info('Position closed @ %.2f; Profit/loss: %.2f%%; Fee: %.2f', close_price, profit_loss,
                 order_info['fee'])

    def cancel_all(self) -> None:
        for txid in self.position['orders']:
            self.exchange.cancel_order(txid)
            self.position['orders'].remove(txid)
