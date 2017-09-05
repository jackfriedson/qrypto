
_base_config = {
    'base_currency': 'ETH',
    'quote_currency': 'USDT',
    'unit': 0.02,
    'ohlc_interval': 30,
    'sleep_duration': 30
}


_config_map = {
    'mfi': {
        'mfi': (14, 80, 20),
        'macd_slope_min': .1
    },
    'tpm': {
        'macd_threshold': 0.3,
        'target_profit': .0225,
        'stop_loss': 0.0075,
    },
    'qlearn': {
        'unit': 1,
        'momentum': (12, 36, 72),
        'train_start': '7/10/2017',
        'train_end': '8/21/2017',
        'fee': 0.
    }
}


def get_config(config_name):
    if config_name not in _config_map:
        raise ValueError('{} is not a valid config name'.format(config_name))
    result = _base_config.copy()
    result.update(_config_map[config_name])
    return result
