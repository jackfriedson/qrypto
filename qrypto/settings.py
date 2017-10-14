
_base_config = {
    'base_currency': 'ETH',
    'quote_currency': 'USD',
    'unit': 0.5,
    'fee': 0.0025,
    'ohlc_interval': 1,
    'sleep_duration': 30,
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
        # 'confidence_thresholds': (0.6, 0.6)
    }
}


def get_config(config_name):
    if config_name not in _config_map:
        raise ValueError('{} is not a valid config name'.format(config_name))
    result = _base_config.copy()
    result.update(_config_map[config_name])
    return result
