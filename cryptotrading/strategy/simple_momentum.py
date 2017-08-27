import logging
import time
from typing import Tuple

from cryptotrading.data.datasets import OHLCDataset
from cryptotrading.data.mixins import MACDMixin
from cryptotrading.strategy.base import BaseStrategy


log = logging.getLogger(__name__)


class TakeProfitMomentumStrategy(BaseStrategy):

    class _Dataset(OHLCDataset, MACDMixin):
        pass

    def __init__(self,
                 exchange,
                 base_currency: str,
                 unit: float,
                 macd_threshold: float,
                 target_profit: float,
                 stop_loss: float,
                 quote_currency:str = 'USD',
                 ohlc_interval: int = 1,
                 sleep_duration: Tuple[int, int] = (30, 60),
                 macd: Tuple[int, int, int] = (10, 26, 9)):
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
        self.take_profit = lambda p: p * (1. + target_profit)
        self.stop_loss = lambda p: p * (1. - stop_loss)

        self.data = self._Dataset(macd=macd)

    def update(self):
        # Get data from exchange
        new_data = self.exchange.get_ohlc(self.base_currency, self.quote_currency,
                                          interval=self.ohlc_interval)
        self.data.update(new_data)
        log.info('{}; {:.2f}'.format(self.data.last, self.data.macd[-1]),
                 extra={'price': self.data.last, 'macd': self.data.macd[-1]})

    def should_open(self):
        return self.data.macd[-1] >= self.macd_threshold

    def should_close(self):
        return self.data.last >= self.take_profit(self.position['open']) \
               or self.data.last <= self.stop_loss(self.position['open'])

    def open_position(self):
        log.info('Opening position...')
        limit_price = self.data.last * 1.001
        order_info = self.exchange.limit_order(self.base_currency, 'buy', limit_price, self.unit,
                                               quote_currency=self.quote_currency)
        open_price = order_info['price']
        self.position = {'open': open_price}
        log.info('Position opened @ %.2f; Fee: %.2f', open_price, order_info['fee'])

    def close_position(self):
        log.info('Closing position...')
        limit_price = self.data.last * 0.999
        order_info = self.exchange.limit_order(self.base_currency, 'sell', limit_price, self.unit,
                                               quote_currency=self.quote_currency)
        close_price = order_info['price']
        profit_loss = 100. * ((close_price / self.position['open']) - 1.)
        self.position = None
        log.info('Position closed @ %.2f; Profit/loss: %.2f%%; Fee: %.2f', close_price, profit_loss,
                 order_info['fee'])
