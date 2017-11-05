from qrypto.data.indicators import BasicIndicator


base_config = {
    'base_currency': 'BTC',
    'quote_currency': 'USD',
    'unit': 1,
    'fee': 0.002,
    'ohlc_interval': 120,
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
        BasicIndicator('adosc'),
        BasicIndicator('stddev'),
        BasicIndicator('tsf'),
        BasicIndicator('linearreg'),
        BasicIndicator('linearreg_slope'),
    ],
    'additional': [
        BasicIndicator('mom', 1),
        BasicIndicator('mom', 6),
        BasicIndicator('mom', 12)
    ]
}


csv_files = [
    # {
    #     'filename': 'avg-block-size.csv',
    #     'name': 'block_size'
    # },
    # {
    #     'filename': 'cost-per-transaction.csv',
    #     'name': 'tx_cost'
    # },
    # {
    #     'filename': 'estimated-transaction-volume-usd.csv',
    #     'name': 'est_tx_vol_usd'
    # },
    # {
    #     'filename': 'hash-rate.csv',
    #     'name': 'hash_rate'
    # },
    # {
    #     'filename': 'median-confirmation-time.csv',
    #     'name': 'confirmation_time'
    # },
    # {
    #     'filename': 'miners-revenue.csv',
    #     'name': 'mining_revenue'
    # },
    # {
    #     'filename': 'trade-volume.csv',
    #     'name': 'trade_vol'
    # },
    # {
    #     'filename': 'transaction-fees.csv',
    #     'name': 'tx_fees'
    # },
    {
        'filename': 'bid_ask_sum.csv',
        'name': ['asks', 'bids'],
        'headers': True
    },
    {
        'filename': 'bid_ask_spread.csv',
        'name': 'bid_ask_spread_pct',
        'headers': True
    }
]

custom_columns = [
    {
        'name': 'bid_ask_difference',
        'inputs': ['asks', 'bids'],
        'func': lambda asks, bids: asks - bids
    }
]


addtl_currencies = ['ETH', 'LTC', 'ETC']


def get_csv_data(csv_dir):
    for csv in csv_files:
        csv['path'] = csv_dir/csv.pop('filename')
    return csv_files, custom_columns


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
    result = base_config.copy()
    result.update(_config_map[config_name])
    return result


no_val_indicators = ['bop', 'trange', 'ht_trendline', 'ad', 'obv', 'ht_dcperiod',
'ht_dcphase', 'ht_phasor', 'ht_sine', 'ht_trendmode', 'mama', 'sar', 'sarext']

single_val_indicators = ['adx', 'adxr', 'aroon', 'aroonosc', 'cci', 'cmo', 'dx',
'mfi', 'minus_di', 'minus_dm', 'mom', 'plus_di', 'plus_dm', 'roc', 'rocp', 'rocr', 'rsi',
'trix', 'willr', 'atr', 'natr', 'dema', 'ema', 'kama', 'midpoint', 'midprice', 'sma',
'tema', 'trima', 'wma', 'beta', 'correl', 'linearreg', 'linearreg_angle', 'linearreg_intercept',
'linearreg_slope', 'stddev', 'tsf', 'var', 'bbands']

multi_val_indicators = {
    'apo': {'fastperiod': 12, 'slowperiod': 26},
    'macd': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
    'ppo': {'fastperiod': 12, 'slowperiod': 26},
    'stoch': {'fastk_period': 5, 'slowk_period': 3, 'slowd_period': 3},
    'stochf': {'fastk_period': 5, 'fastd_period': 3},
    'stochrsi': {'timeperiod': 14, 'fastk_period': 5, 'fastd_period': 3},
    'ultosc': {'timeperiod1': 7, 'timeperiod2': 14, 'timeperiod3': 28},
    'adosc': {'fastperiod': 3, 'slowperiod': 10}
}

periods = [2, 3, 4, 5, 7, 9, 11, 13, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 125, 150, 175]

def get_indicators_full():
    indicators = [
        BasicIndicator(name, period)
        for name in single_val_indicators
        for period in periods
    ]
    indicators.extend([BasicIndicator(name) for name in no_val_indicators])
    for name, config in multi_val_indicators.items():
        for i in range(1, 8):
            scaled_config = {k: v*i for k, v in config.items()}
            indicators.append(BasicIndicator(name, config=scaled_config))
    return indicators

