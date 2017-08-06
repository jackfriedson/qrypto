import time

from cryptotrading.data.ohlc import OHLCDataset
from cryptotrading.strategy.core import BaseStrategy


class TakeProfitMomentumStrategy(BaseStrategy):

    def __init__(self, base_currency, exchange, unit, macd_threshold, target_profit,
                 stop_loss, quote_currency='USD', sleep_duration=60, macd=(10, 26, 9)):
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
        self.take_profit_trigger = '+' + str(target_profit) + '%'
        self.take_profit_limit = '+' + str(target_profit - .25) + '%'
        self.stop_loss_trigger = '-' + str(stop_loss) + '%'
        self.stop_loss_limit = '-' + str(stop_loss - .25) + '%'

        self.data = OHLCDataset(macd=macd)

    def run(self):
        while True:
            self.update()

            # Ensure we only ever have one open position at a time
            if not self.position:
                self.open_conditional()
            else:
                self.close_conditional()

            time.sleep(self.sleep_duration)

    def update(self):
        new_data = self.exchange.recent_ohlc(self.base_currency, self.quote_currency)
        self.data.add_all(new_data)
        print('Price: {}'.format(self.data.last_price()))

    def open_conditional(self):
        macd_data = self.data.macd()
        recent_macd, recent_signal = macd_data[-1]
        macd_difference = recent_macd - recent_signal
        print('MACD: {0:.2f}'.format(macd_difference))
        if macd_difference > self.macd_threshold:
            self.open_position()

    def close_conditional(self):
        for txid, order_info in self.exchange.get_order_info(self.position).items():
            status = order_info['status']
            print('Order {} is {}'.format(txid, status))
            if status in ['closed', 'canceled', 'expired']:
                self.position.remove(txid)
                self.cancel_position()
                self.sleep_duration = 15

    def open_position(self):
        print('Placing market order for {} {} around {}'.format(self.base_currency, self.unit,
                                                                self.data.last_price()))
        order_id = self.exchange.market_order(self.base_currency, True, self.unit)

        # Wait for market order to close
        order_open = True
        while order_open:
            order_info = self.exchange.get_order_info(order_id)
            order_open = order_info['status'] not in ['closed', 'canceled', 'expired']
            if order_open:
                print('Market order still open')
                time.sleep(1)
            else:
                print('Market order closed')

        print('Placing take-profit order at {} with limit {}'.format(self.take_profit_trigger, self.take_profit_limit))
        take_profit = self.exchange.take_profit_limit_order(self.base_currency, False, self.take_profit_trigger,
                                                            self.take_profit_limit, self.unit)
        print('Placing stop-loss order at {} with limit {}'.format(self.stop_loss_trigger, self.stop_loss_limit))
        stop_loss = self.exchange.stop_loss_limit_order(self.base_currency, False, self.stop_loss_trigger,
                                                        self.stop_loss_limit, self.unit)
        self.position.extend([take_profit, stop_loss])
        self.sleep_duration = 30
