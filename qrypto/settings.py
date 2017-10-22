from qrypto.data.indicators import BasicIndicator


_base_config = {
    'base_currency': 'BTC',
    'quote_currency': 'USD',
    'unit': 0.5,
    'fee': 0.0025,
    'ohlc_interval': 5,
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
        'unit': 1
    }
}


qlearn_indicators = {
    'primary': [
        BasicIndicator('rsi', 6),
        BasicIndicator('rsi', 12),
        BasicIndicator('mom', 1),
        BasicIndicator('mom', 3),
        BasicIndicator('obv'),
        BasicIndicator('adx', 14),
        BasicIndicator('adx', 20),
        BasicIndicator('macd'),
        BasicIndicator('bbands'),
        BasicIndicator('willr'),
        BasicIndicator('atr', 14),
        BasicIndicator('rocr', 3),
        BasicIndicator('rocr', 12),
        BasicIndicator('cci', 12),
        BasicIndicator('cci', 20),
        BasicIndicator('sma', 3),
        BasicIndicator('sma', 6),
        BasicIndicator('ema', 6),
        BasicIndicator('ema', 12),
        BasicIndicator('ema', 26),
        BasicIndicator('wma', 6),
        BasicIndicator('mfi', 14),
        BasicIndicator('trix'),
        BasicIndicator('stoch'),
        BasicIndicator('stochrsi'),
        BasicIndicator('ad'),
        BasicIndicator('adosc')
    ],
    'additional': [
        BasicIndicator('mom', 1),
        BasicIndicator('mom', 6),
        BasicIndicator('mom', 12)
    ]
}


csv_files = [
    {
        'filename': 'block_size.csv',
        'name': 'block_size'
    },
    {
        'filename': 'mining_difficulty.csv',
        'name': 'difficulty'
    },
    {
        'filename': 'number_of_transactions.csv',
        'name': 'n_transactions'
    },
    {
        'filename': 'time_between_blocks.csv',
        'name': 'block_rate'
    },
    {
        'filename': 'hash_rate.csv',
        'name': 'hashrate'
    },
    {
        'filename': 'mining_revenue.csv',
        'name': 'revenue'
    }
]


def get_csv_data(csv_dir):
    for csv in csv_files:
        csv['path'] = csv_dir/csv.pop('filename')
    return csv_files


def get_indicators(base, additional):
    result = {}
    result[base] = qlearn_indicators['primary']
    for addtl in additional:
        if addtl != base:
            result[addtl] = qlearn_indicators['additional']
    return result


def get_config(config_name):
    if config_name not in _config_map:
        raise ValueError('{} is not a valid config name'.format(config_name))
    result = _base_config.copy()
    result.update(_config_map[config_name])
    return result
