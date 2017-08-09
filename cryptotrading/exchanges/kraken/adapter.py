import logging
from contextlib import contextmanager
from functools import partial

from cryptotrading.exchanges.errors import APIException, ServiceUnavailableException
from cryptotrading.exchanges.kraken.api import KrakenAPI

log = logging.getLogger(__name__)

currency_map = {
    'BTC': 'XXBT',
    'ETH': 'XETH',
    'USD': 'ZUSD',
}


@contextmanager
def handle_api_exception():
    try:
        yield
    except ServiceUnavailableException as e:
        # TODO: retry
        pass
    except APIException as e:
        log.exception('Kraken returned an error -- %s', str(e))
        raise

class KrakenAPIAdapter(object):
    """ Adapter from the core Kraken API to the exchange API interface.
    """

    # TODO: Add session functionality (reuse existing HTTP connection)

    def __init__(self, api=None, key=None, secret=None, key_path=None):
        if api and issubclass(api, KrakenAPI):
            self.api = api
        elif key_path:
            self.api = KrakenAPI()
            self.api.load_api_key(key_path)
        else:
            self.api = KrakenAPI(key=key, secret=secret)

        self.order_method = self._place_order_with_logging
        self.last_txs = {}

    def _place_order_with_logging(self, pair, buy_sell, order_type, volume, **kwargs):
        order_data = {
            'pair': pair,
            'buy_sell': buy_sell,
            'order_type': order_type,
            'volume': volume
        }
        order_data.update(kwargs)

        log.info('%s order placed', order_type, extra={
            'event_name': 'order_open',
            'event_data': order_data
        })
        return self.api.add_standard_order(pair, buy_sell, order_type, volume, **kwargs)

    @classmethod
    def _generate_currency_pair(cls, base, quote):
        base = currency_map.get(base, base)
        quote = currency_map.get(quote, quote)
        return base + quote

    def _get_data_since_last(self, data_method, name, base_currency, quote_currency, **kwargs):
        since = self.last_txs.get(name)
        pair = self._generate_currency_pair(base_currency, quote_currency)
        resp = data_method(pair, since=since, **kwargs)
        self.last_txs[name] = resp['last']
        return resp[pair]

    def _partial_order(self, base_currency, quote_currency, buy_sell):
        pair = self._generate_currency_pair(base_currency, quote_currency)
        return partial(self.order_method, pair, buy_sell)

    # Market Info
    @handle_api_exception()
    def order_book(self, base_currency, quote_currency='USD'):
        """
        :returns: {
            'asks': list of asks,
            'bids': list of bids
        }
        """
        pair = self._generate_currency_pair(base_currency, quote_currency)
        resp = self.api.get_order_book(pair)
        return resp[pair]

    @handle_api_exception()
    def recent_trades(self, base_currency, quote_currency='USD'):
        data = self._get_data_since_last(self.api.get_recent_trades, 'trades', base_currency, quote_currency)
        return [{
            'price': t[0],
            'volume': t[1],
            'time': t[2],
            'buy_sell': t[3],
            'type': t[4],
            'misc': t[5]
        } for t in data]

    @handle_api_exception()
    def recent_ohlc(self, base_currency, quote_currency='USD', interval=1):
        """
        :param interval: time period duration in minutes (see KrakenAPI for valid intervals)
        :returns: {
            'time': ,
            'open': opening price for the time period,
            'high': highest price for the time period,
            'low': lowest price for the time period,
            'close': closing price for the time period,
            'vwap': volume weighted average price,
            'volume': ,
            'count': ,
        }
        """
        data = self._get_data_since_last(self.api.get_OHLC_data, 'ohlc', base_currency, quote_currency,
                                         interval=interval)
        return [{
            'time': d[0],
            'open': d[1],
            'high': d[2],
            'low': d[3],
            'close': d[4],
            'vwap': d[5],
            'volume': d[6],
            'count': d[7]
        } for d in data]

    @handle_api_exception()
    def recent_spread(self, base_currency, quote_currency='USD'):
        data = self._get_data_since_last(self.api.get_recent_spread_data, 'spread', base_currency, quote_currency)
        return [{
            'time': s[0],
            'bid': s[1],
            'ask': s[2]
        } for s in data]

    # User Info
    @handle_api_exception()
    def get_orders_info(self, txids):
        """
        :returns: {
            txid: {order info}
        }
        """
        txid_string = ','.join(txids)
        resp = self.api.query_orders_info(txid_string)
        return resp

    # Orders
    @handle_api_exception()
    def market_order(self, base_currency, buy_sell, volume, quote_currency='USD'):
        """
        :returns: txid of the placed order
        """
        order_fn = self._partial_order(base_currency, quote_currency, buy_sell)
        resp = order_fn('market', volume)
        return resp['txid']

    @handle_api_exception()
    def limit_order(self, base_currency, buy_sell, price, volume, quote_currency='USD'):
        order_fn = self._partial_order(base_currency, quote_currency, buy_sell)
        resp = order_fn('limit', volume, price=price)
        return resp['txid']

    @handle_api_exception()
    def stop_loss_order(self, base_currency, buy_sell, price, volume, quote_currency='USD'):
        order_fn = self._partial_order(base_currency, quote_currency, buy_sell)
        resp = order_fn('stop-loss', volume, price=price)
        return resp['txid']

    @handle_api_exception()
    def stop_loss_limit_order(self, base_currency, buy_sell, stop_loss_price, limit_price, volume,
                              quote_currency='USD'):
        order_fn = self._partial_order(base_currency, quote_currency, buy_sell)
        resp = order_fn('stop-loss-limit', volume, price=stop_loss_price, price2=limit_price)
        return resp['txid']

    @handle_api_exception()
    def take_profit_order(self, base_currency, buy_sell, price, volume, quote_currency='USD'):
        order_fn = self._partial_order(base_currency, quote_currency, buy_sell)
        resp = order_fn('take-profit', volume, price=price)
        return resp['txid']

    @handle_api_exception()
    def take_profit_limit_order(self, base_currency, buy_sell, take_profit_price, limit_price, volume,
                                quote_currency='USD'):
        order_fn = self._partial_order(base_currency, quote_currency, buy_sell)
        resp = order_fn('take-profit-limit', volume, price=take_profit_price, price2=limit_price)
        return resp['txid']

    @handle_api_exception()
    def trailing_stop_order(self, base_currency, buy_sell, trailing_stop_offset, volume, quote_currency='USD'):
        order_fn = self._partial_order(base_currency, quote_currency, buy_sell)
        resp = order_fn('trailing-stop', volume, price=trailing_stop_offset)
        return resp['txid']

    @handle_api_exception()
    def trailing_stop_limit_order(self, base_currency, buy_sell, trailing_stop_offset, limit_offset,
                                  volume, quote_currency='USD'):
        order_fn = self._partial_order(base_currency, quote_currency, buy_sell)
        resp = order_fn('trailing-stop-limit', volume, price=trailing_stop_offset, price2=limit_offset)
        return resp['txid']

    def cancel_order(self, order_id):
        log.info('Cancelling order %s', order_id,
                 extra={
                     'event_name': 'cancel_order',
                     'event_data': {'order_id': order_id}
                 })
        self.api.cancel_open_order(txid=order_id)
