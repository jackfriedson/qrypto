import logging
import time

from cryptotrading.data.datasets import OHLCDataset
from cryptotrading.data.mixins import MACDMixin
from cryptotrading.strategy.base import BaseStrategy


log = logging.getLogger(__name__)


class _Dataset(MACDMixin, OHLCDataset):
    pass


class TakeProfitMomentumStrategy(BaseStrategy):

    def __init__(self,
                 base_currency,
                 exchange,
                 unit,
                 macd_threshold,
                 target_profit,
                 stop_loss,
                 buffer_percent=0.0025,
                 quote_currency='USD',
                 sleep_duration=(15, 30),
                 macd=(10, 26, 9)):
        """
        :param base_currency:
        :param exchange:
        :param unit: volume of base_currency to buy every time a position is opened
        :type unit: float
        :param macd_threshold:
        :param target_profit:
        :param stop_loss_trigger:
        :param stop_loss_limit:
        :param quote_currency:
        """
        super(TakeProfitMomentumStrategy, self).__init__(base_currency, exchange, unit, quote_currency,
                                                         sleep_duration)

        self.macd_threshold = macd_threshold
        self.take_profit_trigger = lambda p: p * (1. + target_profit)
        self.take_profit_limit = lambda p: p * (1. + (target_profit - buffer_percent))
        self.stop_loss_trigger = lambda p: p * (1. - stop_loss)
        self.stop_loss_limit = lambda p: p * (1. - (stop_loss + buffer_percent))

        self.data = _Dataset(macd_values=macd)

    def update(self):
        # Get data from exchange
        new_data = self.exchange.recent_ohlc(self.base_currency, self.quote_currency)
        self.data.add_all(new_data)

        # Calculate MACD
        macd_history = self.data.macd()
        last_macd, signal = macd_history[-1]
        self.indicators['macd'] = last_macd - signal
        last_price = self.data.last_price()

        log.info('{}; {:.2f}'.format(last_price, self.indicators['macd']),
                 extra={'price': last_price, 'macd': self.indicators['macd']})

    def should_open(self):
        # TODO: Account for slope of increasing MACD
        result = self.indicators['macd'] >= self.macd_threshold
        if result:
            log.info('Opening position...')
        return result

    def should_close(self):
        return self.any_orders_closed()

    def open_position(self):
        txids = self.exchange.market_order(self.base_currency, 'buy', self.unit)
        market_order_info = self.wait_for_order_close(txids[0])
        market_price = float(market_order_info['price'])

        take_profit_ids = self.exchange.take_profit_limit_order(self.base_currency, 'sell',
                                                                self.take_profit_trigger(market_price),
                                                                self.take_profit_limit(market_price),
                                                                self.unit)
        stop_loss_ids = self.exchange.stop_loss_limit_order(self.base_currency, 'sell',
                                                            self.stop_loss_trigger(market_price),
                                                            self.stop_loss_limit(market_price), self.unit)
        self.positions.extend(take_profit_ids + stop_loss_ids)

        log.info('Position opened: Bought @ %f; Selling @ %f or %f',
                 market_price, self.take_profit_trigger(market_price), self.stop_loss_trigger(market_price))

    def close_position(self):
        self.cancel_all()

    def any_orders_closed(self):
        """ Checks if any open positions have been closed.

        Returns true if any open positions are closed, canceled, or expired, and removes
        those positions from the list.
        """
        orders = self.exchange.get_orders_info(self.positions)
        for txid, order_info in orders.items():
            if order_info['status'] == 'closed':
                log.info('Position closed at %s', order_info['cost'])
                self.positions.remove(txid)
                return True
        return False

    def wait_for_order_close(self, txid, sleep_inverval=2):
        order_open = True
        while order_open:
            time.sleep(sleep_inverval)
            order_info = self.exchange.get_order_info(txid)
            order_open = order_info['status'] not in ['closed', 'canceled', 'expired']
        return order_info