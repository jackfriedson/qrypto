from cryptotrading.exceptions import CryptoTradingException

class APIException(CryptoTradingException):
    pass

class AuthorizationException(APIException):
    pass
