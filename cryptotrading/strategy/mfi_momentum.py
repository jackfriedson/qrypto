import logging
import time
from typing import Tuple

from cryptotrading.data.datasets import OHLCDataset
from cryptotrading.data.mixins import MACDMixin, MFIMixin
from cryptotrading.strategy.base import BaseStrategy


log = logging.getLogger(__name__)


class _Dataset(MACDMixin, MFIMixin, OHLCDataset):
    pass


class MFIMomentumStrategy(BaseStrategy):

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

        self.data = _Dataset(macd=macd, mfi=mfi_period)
        self.indicators['mfi'] = []

    def update(self):
        new_data = self.exchange.get_ohlc(self.base_currency, self.quote_currency,
                                          interval=self.ohlc_interval)
        self.data.add_all(new_data)
        self.indicators['macd_slope'] = self.data.macd_slope()
        self.indicators['mfi'] = self.data.mfi()

        print(self.data._data.ix[-5:, ['close', 'volume']])
        log.info('%.2f; %.2f; %.2f', self.data.last, self.indicators['mfi'][-1], self.indicators['macd_slope'])

    def should_open(self):
        return len(self.indicators['mfi']) >= 2 \
               and self.indicators['mfi'][-2] < self.mfi_bottom \
               and self.indicators['mfi'][-1] >= self.mfi_bottom \
               and self.indicators['macd_slope'] > self.macd_slope_min

    def should_close(self):
        return (self.indicators['mfi'][-2] > self.mfi_top \
               and self.indicators['mfi'][-1] <= self.mfi_top) \
               or self.data.last < self.position['stop_limit']

    def open_position(self):
        log.info('Opening position...')
        txids = self.exchange.market_order(self.base_currency, 'buy', self.unit)
        market_order_info = self.wait_for_order_close(txids[0])
        open_price = market_order_info['price']
        self.position = {
            'open': open_price,
            'stop_limit': open_price * (1. - self.stop_loss)
        }
        log.info('Position opened @ %.2f', open_price)

    def close_position(self):
        log.info('Closing position...')
        txids = self.exchange.market_order(self.base_currency, 'sell', self.unit)
        market_order_info = self.wait_for_order_close(txids[0])
        close_price = market_order_info['price']
        profit_loss = 100. * ((close_price / self.position['open']) - 1.)
        log.info('Position closed @ %.2f; Profit/loss: %.2f%%', close_price, profit_loss)
        self.position = None
