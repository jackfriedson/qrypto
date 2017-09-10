
_base_config = {
    'base_currency': 'ETH',
    'quote_currency': 'USDT',
    'unit': 0.02,
    'ohlc_interval': 5,
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
        'train_start': '6/1/2017',
        'train_end': '7/1/2017',
        'fee': 0.
    }
}


def get_config(config_name):
    if config_name not in _config_map:
        raise ValueError('{} is not a valid config name'.format(config_name))
    result = _base_config.copy()
    result.update(_config_map[config_name])
    return result
