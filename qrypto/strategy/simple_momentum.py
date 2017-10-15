import logging
import time
from typing import Tuple

from qrypto.data.datasets import OHLCDataset
from qrypto.data.indicators import BasicIndicator
from qrypto.strategy.base import BaseStrategy


log = logging.getLogger(__name__)


class TakeProfitMomentumStrategy(BaseStrategy):

    def __init__(self,
                 exchange,
                 base_currency: str,
                 unit: float,
                 macd_threshold: float,
                 target_profit: float,
                 stop_loss: float,
                 quote_currency: str = 'USD',
                 ohlc_interval: int = 5,
                 sleep_duration: int = 30,
                 macd: Tuple[int, int, int] = (10, 26, 9)) -> None:
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
        indicators = [BasicIndicator('macd')]
        self.data = OHLCDataset(indicators)
        self.ohlc_interval = ohlc_interval
        self.macd_threshold = macd_threshold
        self.take_profit = lambda p: p * (1. + target_profit)
        self.stop_loss = lambda p: p * (1. - stop_loss)

    def update(self) -> None:
        new_data = self.exchange.get_ohlc(self.base_currency, self.quote_currency,
                                          interval=self.ohlc_interval)
        self.data.update(new_data)
        log.info('{}; {:.2f}'.format(self.data.last_price, self.data.macd[-1]),
                 extra={'price': self.data.last_price, 'macd': self.data.macd[-1]})

    def should_open(self) -> bool:
        return self.data.macd[-1] >= self.macd_threshold

    def should_close(self) -> bool:
        return self.data.last_price >= self.take_profit(self.position['open']) \
               or self.data.last_price <= self.stop_loss(self.position['open'])
