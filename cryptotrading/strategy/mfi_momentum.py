import logging
import time
from typing import Tuple

from cryptotrading.data.datasets import OHLCDataset
from cryptotrading.data.mixins import MACDMixin, MFIMixin
from cryptotrading.strategy.base import BaseStrategy


log = logging.getLogger(__name__)


class MFIMomentumStrategy(BaseStrategy):

    class _Dataset(OHLCDataset, MACDMixin, MFIMixin):
        pass

    def __init__(self,
                 base_currency: str,
                 exchange,
                 unit: float,
                 quote_currency: str = 'USD',
                 ohlc_interval: int = 60,
                 stop_loss: float = 0.05,
                 sleep_duration: int = 30*60,
                 macd: Tuple[int, int, int] = (10, 26, 9),
                 macd_slope_min: float = 0.0,
                 mfi: Tuple[int, int, int] = (14, 80, 20)):
        super(MFIMomentumStrategy, self).__init__(base_currency, exchange, unit, quote_currency,
                                                  sleep_duration)
        self.ohlc_interval = ohlc_interval
        self.stop_loss = stop_loss
        self.macd_slope_min = macd_slope_min
        mfi_period, self.mfi_top, self.mfi_bottom = mfi
        self.data = self._Dataset(macd=macd, mfi=mfi_period)

    def update(self):
        new_data = self.exchange.get_ohlc(self.base_currency, self.quote_currency,
                                          interval=self.ohlc_interval, since_last=True)
        self.data.update(new_data)

        print(self.data._data.tail(5)[['close', 'volume', 'mfi']])
        log.info('%.2f; %.2f; %.2f', self.data.last, self.data.mfi[-1], self.data.macd_slope())

    def should_open(self):
        return len(self.data.mfi) >= 2 \
               and self.data.mfi[-2] < self.mfi_bottom \
               and self.data.mfi[-1] >= self.mfi_bottom \
               and self.data.macd_slope() > self.macd_slope_min

    def should_close(self):
        return (self.data.mfi[-2] > self.mfi_top \
               and self.data.mfi[-1] <= self.mfi_top) \
               or self.data.last < self.position['stop_limit']

    def open_position(self):
        log.info('Opening position...')
        limit_price = self.data.last * 1.001
        order_info = self.exchange.limit_order(self.base_currency, 'buy', limit_price, self.unit,
                                               quote_currency=self.quote_currency)
        open_price = order_info['price']
        self.position = {
            'open': open_price,
            'stop_limit': open_price * (1. - self.stop_loss)
        }
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
