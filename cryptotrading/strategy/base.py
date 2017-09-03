"""
"""
import logging
import time
from typing import Tuple


log = logging.getLogger(__name__)


class BaseStrategy(object):

    def __init__(self, base_currency: str, exchange, unit: float, quote_currency: str = 'USD',
                 sleep_duration: int = 60, **kwargs):
        """
        """
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.exchange = exchange
        self.unit = unit
        self.data = None
        self.position = None
        self.sleep_duration = sleep_duration

    def strategy_iter(self):
        self.update()
        if not self.position and self.should_open():
            self.open_position()
        elif self.position and self.should_close():
            self.close_position()

    def run(self) -> None:
        while True:
            self.strategy_iter()
            time.sleep(self.sleep_duration)

    def test(self, test_exchange):
        self.exchange = test_exchange

        keep_running = True
        while keep_running:
            try:
                self.strategy_iter()
            except StopIteration:
                keep_running = False

        self.exchange.print_results()
        self.data.plot()

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
        self.data.add_order('buy', order_info)
        open_price = order_info['price']
        self.position = {'open': open_price}
        log.info('Position opened @ %.2f; Fee: %.2f', open_price, order_info['fee'])

    def close_position(self) -> None:
        log.info('Closing position...')
        # TODO: use last bid instead of 0.1%
        limit_price = self.data.last * 0.999
        order_info = self.exchange.limit_order(self.base_currency, 'sell', limit_price, self.unit,
                                               quote_currency=self.quote_currency)
        self.data.add_order('sell', order_info)
        close_price = order_info['price']
        profit_loss = 100. * ((close_price / self.position['open']) - 1.)
        self.position = None
        log.info('Position closed @ %.2f; Profit/loss: %.2f%%; Fee: %.2f', close_price, profit_loss,
                 order_info['fee'])

    def cancel_all(self) -> None:
        for txid in self.position['orders']:
            self.exchange.cancel_order(txid)
            self.position['orders'].remove(txid)
