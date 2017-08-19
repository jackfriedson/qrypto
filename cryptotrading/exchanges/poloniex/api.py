from src import PoloniexAPI

class PoloniexAdapter(object):

    def __init__(self, key_path: str) -> None:
        apikey = self.load_api_key(key_path)
        self.api = PoloniexAPI(apikey)

    @classmethod
    def load_api_key(cls, path: str) -> dict():
        with open(path, 'r') as f:
            key = f.readline().strip()
            secret = f.readline().strip()
        return {
            'api_key': key,
            'secret': secret
        }

    @classmethod
    def pair(cls, base_currency: str, quote_currency: str) -> str:
        return quote_currency + '_' + base_currency

    def get_ohlc(self, base_currency: str, quote_currency: str = 'USDT', interval: int = 5) -> dict():
        """
        """
        currency_pair = self.pair(base_currency, quote_currency)
        interval *= 60
        result = self.api.returnChartData(currency_pair, period=interval)
        return result

