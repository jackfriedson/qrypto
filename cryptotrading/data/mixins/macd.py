from cryptotrading.data.utils import ema

class MACDMixin(object):

    # TODO: Cache MACD data instead of recomputing every time

    def __init__(self, *args, **kwargs):
        """
        :param macd_values:
        :type macd_values: tuple
        """
        self.n_short_ema, self.n_long_ema, self.n_signal_ema = kwargs.pop('macd_values')
        super(MACDMixin, self).__init__(*args, **kwargs)

    def macd(self):
        """
        :returns: [(macd, signal), ...]
        """
        # TODO: Limit amount of history data in MACD calculation

        # Default to close value if no trades for that period
        avg_price_data = [d['vwap'] if float(d['volume']) > 0 else d['close'] for _, d in self.items()]
        fast = ema(avg_price_data, self.n_short_ema)
        slow = ema(avg_price_data, self.n_long_ema)
        macd = fast - slow
        signal = ema(macd, self.n_signal_ema)
        return list(zip(macd, signal))
