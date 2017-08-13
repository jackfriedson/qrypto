import logging
from contextlib import contextmanager

from cryptotrading.exchanges.errors import APIException
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
    except APIException as e:
        log.exception('Kraken returned an error -- %s', str(e), extra={
            'event_name': 'kraken_error',
            'event_data': {}
        })
        raise


class KrakenAPIAdapter(object):
    """ Adapter from the core Kraken API to the exchange API interface.
    """

    def __init__(self, api=None, key=None, secret=None, key_path=None):
        if api and issubclass(api, KrakenAPI):
            self.api = api
        elif key_path:
            self.api = KrakenAPI()
            self.api.load_api_key(key_path)
        else:
            self.api = KrakenAPI(key=key, secret=secret)

        self.last_txs = {}

    @handle_api_exception()
    def _place_order(self,
                     order_type,
                     base_currency,
                     buy_sell,
                     volume,
                     quote_currency='USD',
                     **kwargs):
        """
        """

        pair = self._generate_currency_pair(base_currency, quote_currency)
        resp = self.api.add_standard_order(pair, buy_sell, order_type, volume, **kwargs)

        txid = resp['txid']
        order_data = {
            'txid': txid,
            'pair': pair,
            'type': buy_sell,
            'order_type': order_type,
            'volume': volume
        }
        order_data.update(kwargs)
        log.info('%s order placed: %s', order_type, txid, extra={
            'event_name': 'order_open',
            'event_data': order_data
        })

        return txid

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
            'avg': volume weighted average price,
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
            'avg': d[5],
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

        result = {}
        for txid, info in resp.items():
            order_info = {
                'txid': txid,
                'status': info['status'],
                'cost': info['cost'],
                'price': info['price'],
                'volume': info['vol'],
                'fee': info['fee']
            }
            result[txid] = order_info

            # Log order close
            status = info['status']
            if status in ['closed', 'canceled', 'expired']:
                log.info('Got info on order %s', txid, extra={
                    'event_name': 'order_' + status,
                    'event_data': order_info
                })

        return result

    def get_order_info(self, txid):
        return self.get_orders_info([txid]).get(txid)

    # Orders
    def market_order(self, base_currency, buy_sell, volume, **kwargs):
        """
        :returns: txid of the placed order
        """
        return self._place_order('market', base_currency, buy_sell, volume, **kwargs)

    def limit_order(self, base_currency, buy_sell, price, volume, **kwargs):
        return self._place_order('limit', base_currency, buy_sell, volume, price=price, **kwargs)

    def stop_loss_order(self, base_currency, buy_sell, price, volume, **kwargs):
        return self._place_order('stop-loss', base_currency, buy_sell, volume, price=price, **kwargs)

    def stop_loss_limit_order(self, base_currency, buy_sell, stop_loss_price, limit_price, volume, **kwargs):
        return self._place_order('stop-loss-limit', base_currency, buy_sell, volume, price=stop_loss_price,
                                 price2=limit_price, **kwargs)

    def take_profit_order(self, base_currency, buy_sell, price, volume, **kwargs):
        return self._place_order('take-profit', base_currency, buy_sell, volume, price=price, **kwargs)

    def take_profit_limit_order(self, base_currency, buy_sell, take_profit_price, limit_price, volume, **kwargs):
        return self._place_order('take-profit-limit', base_currency, buy_sell, volume,
                                 price=take_profit_price, price2=limit_price, **kwargs)

    def trailing_stop_order(self, base_currency, buy_sell, trailing_stop_offset, volume, **kwargs):
        return self._place_order('trailing-stop', base_currency, buy_sell, volume,
                                 price=trailing_stop_offset, **kwargs)

    def trailing_stop_limit_order(self, base_currency, buy_sell, trailing_stop_offset, limit_offset,
                                  volume, **kwargs):
        return self._place_order('trailing-stop-limit', base_currency, buy_sell, volume,
                                 price=trailing_stop_offset, price2=limit_offset, **kwargs)

    def cancel_order(self, order_id):
        resp = self.api.cancel_open_order(txid=order_id)
        log.info('Canceled order %s', order_id, extra={
            'event_name': 'cancel_order',
            'event_data': resp
        })
