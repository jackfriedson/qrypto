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
                 ohlc_interval=1,
                 sleep_duration=(30, 60),
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
        self.ohlc_interval = ohlc_interval

        self.macd_threshold = macd_threshold
        self.take_profit_trigger = lambda p: p * (1. + target_profit)
        self.take_profit_limit = lambda p: p * (1. + (target_profit - buffer_percent))
        self.stop_loss_trigger = lambda p: p * (1. - stop_loss)
        self.stop_loss_limit = lambda p: p * (1. - (stop_loss + buffer_percent))

        self.data = _Dataset(macd=macd)

    def update(self):
        # Get data from exchange
        new_data = self.exchange.recent_ohlc(self.base_currency, self.quote_currency,
                                             interval=self.ohlc_interval)
        self.data.add_all(new_data)
        _, _, macdhist = self.data.macd()
        self.indicators['macd'] = macdhist[-1]
        last = self.data.last

        log.info('{}; {:.2f}'.format(last, self.indicators['macd']),
                 extra={'price': last, 'macd': self.indicators['macd']})

    def should_open(self):
        return self.indicators['macd'] >= self.macd_threshold

    def should_close(self):
        return self.any_orders_closed()

    def open_position(self):
        log.info('Opening position...')
        txids = self.exchange.market_order(self.base_currency, 'buy', self.unit)
        market_order_info = self.wait_for_order_close(txids[0])
        open_price = float(market_order_info['price'])
        take_profit_ids = self.exchange.take_profit_limit_order(self.base_currency, 'sell',
                                                                self.take_profit_trigger(open_price),
                                                                self.take_profit_limit(open_price),
                                                                self.unit)
        stop_loss_ids = self.exchange.stop_loss_limit_order(self.base_currency, 'sell',
                                                            self.stop_loss_trigger(open_price),
                                                            self.stop_loss_limit(open_price), self.unit)
        self.position = {
            'open': open_price,
            'orders': take_profit_ids + stop_loss_ids
        }
        log.info('Position opened: Bought @ {:.2f}; Selling @ {:.2f} or {:.2f}'.format(
            open_price, self.take_profit_trigger(open_price), self.stop_loss_trigger(open_price)))

    def close_position(self):
        log.info('Closing position...')
        self.cancel_all()
        self.position = None
