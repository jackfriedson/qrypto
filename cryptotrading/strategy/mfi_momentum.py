import logging
import time
from collections import OrderedDict

from cryptotrading.data.datasets import OHLCDataset
from cryptotrading.data.mixins import MACDMixin, MFIMixin
from cryptotrading.strategy.base import BaseStrategy


log = logging.getLogger(__name__)


class _Dataset(MACDMixin, MFIMixin, OHLCDataset):
    pass


class MFIMomentumStrategy(BaseStrategy):

    def __init__(self,
                 base_currency,
                 exchange,
                 unit,
                 quote_currency='USD',
                 ohlc_interval=60,
                 stop_loss=0.05,
                 sleep_duration=30*60,
                 macd=(10, 26, 9),
                 macd_slope_min=0.0,
                 mfi=(14, 80, 20)):
        """
        """
        super(MFIMomentumStrategy, self).__init__(base_currency, exchange, unit, quote_currency,
                                                  sleep_duration)

        self.ohlc_interval = ohlc_interval
        self.macd_slope_min = macd_slope_min
        mfi_period, self.mfi_top, self.mfi_bottom = mfi

        self.data = _Dataset(macd=macd, mfi=mfi_period)
        self.indicators['mfi'] = []
        self.last_timestamp = None

    def update(self):
        new_data = self.exchange.recent_ohlc(self.base_currency, self.quote_currency,
                                             interval=self.ohlc_interval)
        self.data.add_all(new_data)
        self.indicators['macd_slope'] = self.data.macd_slope()

        mfi = self.data.mfi()
        if not self.last_timestamp or self.data.last_timestamp > self.last_timestamp:
            # Only add data if this is a new OHLC interval
            self.indicators['mfi'].append(mfi)
            self.last_timestamp = self.data.last_timestamp
        else:
            # If same interval, just update existing data
            self.indicators['mfi'][-1] = mfi

    def should_open(self):
        return len(self.indicators['mfi']) >= 2 \
               # MFI has crossed above the bottom threshold
               and self.indicators['mfi'][-2] < self.mfi_bottom \
               and self.indicators['mfi'][-1] >= self.mfi_bottom \
               # MACD is increasing
               and self.indicators['macd_slope'] > self.macd_slope_min

    def should_close(self):
        return self.indicators['mfi'][-2] > self.mfi_top \
               and self.indicators['mfi'][-1] <= self.mfi_top \
               # Essentially a stop loss to ensure we don't lose more than 5%
               or self.data.last < self.position.purchase_price * (1. - self.stop_loss)

    def open_position(self):
        pass

    def close_position(self):
        pass
