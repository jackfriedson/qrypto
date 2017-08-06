#!/usr/local/bin/python3

import os
import pprint

from cryptotrading.exchanges import Kraken
from cryptotrading.exchanges.kraken.api import KrakenAPI
from cryptotrading.strategy.momentum import TakeProfitMomentumStrategy

printer = pprint.PrettyPrinter(indent=1)

config = {
    'unit': 0.02,
    'macd_threshold': 0.2,
    'target_profit': 2.5,
    'stop_loss': 0.5,
    'sleep_duration': (15,30)
}

keypath = os.path.expanduser('~/.kraken_api_key')
kraken = Kraken(key_path=keypath)
strategy = TakeProfitMomentumStrategy('ETH', kraken, **config)
strategy.run()
