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
        BasicIndicator('beta'),
        BasicIndicator('linearreg'),
        BasicIndicator('linearreg_angle'),
        BasicIndicator('linearreg_intercept'),
        BasicIndicator('linearreg_slope'),
        BasicIndicator('tsf'),
        BasicIndicator('var'),
    ],
    'additional': [
        BasicIndicator('mom', 1),
        BasicIndicator('mom', 6),
        BasicIndicator('mom', 12)
    ]
}


csv_files = [
    {
        'filename': 'avg-block-size.csv',
        'name': 'block_size'
    },
    {
        'filename': 'cost-per-transaction.csv',
        'name': 'tx_cost'
    },
    {
        'filename': 'estimated-transaction-volume-usd.csv',
        'name': 'est_tx_vol_usd'
    },
    {
        'filename': 'hash-rate.csv',
        'name': 'hash_rate'
    },
    {
        'filename': 'median-confirmation-time.csv',
        'name': 'confirmation_time'
    },
    {
        'filename': 'miners-revenue.csv',
        'name': 'mining_revenue'
    },
    {
        'filename': 'trade-volume.csv',
        'name': 'trade_vol'
    },
    {
        'filename': 'transaction-fees.csv',
        'name': 'tx_fees'
    },
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
