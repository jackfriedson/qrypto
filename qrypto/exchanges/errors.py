from qrypto.errors import CryptoTradingException

class APIException(CryptoTradingException):
    pass

class AuthorizationException(APIException):
    pass

class ServiceUnavailableException(APIException):
    pass
