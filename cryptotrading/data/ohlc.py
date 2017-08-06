from collections import OrderedDict

from cryptotrading.data.utils import ema

class OHLCDataset(object):

    def __init__(self, max_history=10000, macd=None):
        """
        :param max_history:
        :param macd: tuple of the form (<>, <>, <>)
        """
        self.max_history = max_history
        self.last_timestamp = None
        self.data = OrderedDict()

        if macd:
            self.fast_ema, self.slow_ema, self.signal_line = macd

    def add_all(self, incoming_data):
        """ Expects each data point to be an array formatted as follows:
                [<time>, <open>, <high>, <low>, <close>, <vwap>, <volume>, <count>]
        """
        for entry in incoming_data:
            self.data[entry[0]] = {
                'open': entry[1],
                'high': entry[2],
                'low': entry[3],
                'close': entry[4],
                'vwap': entry[5],
                'volume': entry[6],
                'count': entry[7]
            }

            if len(self.data) >= self.max_history:
                self.data.popitem(last=False)

        self.last_timestamp = list(self.data.keys())[-1]

    def last_price(self):
        return self.data[self.last_timestamp].get('close')

    def macd(self):
        """
        :returns: [(macd, signal), ...]
        """
        vwap_data = [d['close'] for _, d in self.data.items()]
        fast = ema(vwap_data, self.fast_ema)
        slow = ema(vwap_data, self.slow_ema)
        macd = fast - slow
        signal = ema(macd, self.signal_line)
        return list(zip(macd, signal))
