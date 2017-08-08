import logging
import os

import click

from cryptotrading.exchanges import Kraken
from cryptotrading.strategy.momentum import TakeProfitMomentumStrategy


config = {
    'unit': 0.05,
    'macd_threshold': 0.3,
    'target_profit': 2.5,
    'stop_loss': 0.5,
    'sleep_duration': (15,30)
}


@click.command()
def cli():
    keypath = os.path.expanduser('~/.kraken_api_key')
    kraken = Kraken(key_path=keypath)
    strategy = TakeProfitMomentumStrategy('ETH', kraken, **config)
    strategy.run()
