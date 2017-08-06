import time

from cryptotrading.data.datasets import OHLCDataset
from cryptotrading.data.mixins import MACDMixin
from cryptotrading.strategy.base import BaseStrategy


class _Dataset(MACDMixin, OHLCDataset):
    pass


class TakeProfitMomentumStrategy(BaseStrategy):

    def __init__(self, base_currency, exchange, unit, macd_threshold, target_profit,
                 stop_loss, buffer_percent=0.25, quote_currency='USD', sleep_duration=(15, 30),
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
        self.take_profit_trigger = '+{}%'.format(target_profit)
        self.take_profit_limit = '+{}%'.format(target_profit - buffer_percent)
        self.stop_loss_trigger = '-{}%'.format(stop_loss)
        self.stop_loss_limit = '-{}%'.format(stop_loss - buffer_percent)

        self.data = _Dataset(macd_values=macd)

    def update(self):
        new_data = self.exchange.recent_ohlc(self.base_currency, self.quote_currency)
        self.data.add_all(new_data)
        print('Price: {}'.format(self.data.last_price()))

    def open_condition(self):
        macd_history = self.data.macd()
        macd, signal = macd_history[-1]
        macd_difference = macd - signal
        print('MACD: {0:.2f}'.format(macd_difference))
        if macd_difference > self.macd_threshold:
            self.open_position()

    def close_condition(self):
        self.cancel_all_if_any_close()

    def open_position(self):
        print('Placing market order for {} {} around {}'.format(self.base_currency, self.unit,
                                                                self.data.last_price()))
        txids = self.exchange.market_order(self.base_currency, 'buy', self.unit)

        # Wait for market order to close
        order_open = True
        while order_open:
            time.sleep(2)
            order_info = self.exchange.get_orders_info(txids).get(txids[0])
            order_open = order_info['status'] not in ['closed', 'canceled', 'expired']
            if order_open:
                print('Market order still open')
            else:
                print('Market order closed')

        print('Placing take-profit order at {} with limit {}'.format(self.take_profit_trigger, self.take_profit_limit))
        take_profit_ids = self.exchange.take_profit_limit_order(self.base_currency, 'sell', self.take_profit_trigger,
                                                            self.take_profit_limit, self.unit)
        self.positions.extend(take_profit_ids)
        print('Placing stop-loss order at {} with limit {}'.format(self.stop_loss_trigger, self.stop_loss_limit))
        stop_loss_ids = self.exchange.stop_loss_limit_order(self.base_currency, 'sell', self.stop_loss_trigger,
                                                        self.stop_loss_limit, self.unit)
        self.positions.extend(stop_loss_ids)
