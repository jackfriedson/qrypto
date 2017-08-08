import os
from logging.config import fileConfig

import click

from cryptotrading.exchanges import Kraken
from cryptotrading.strategy.momentum import TakeProfitMomentumStrategy


API_KEY = os.path.expanduser('~/.kraken_api_key')
LOG_CONFIG = 'cryptotrading/log_config.ini'


config = {
    'unit': 0.05,
    'macd_threshold': 0.3,
    'target_profit': 2.5,
    'stop_loss': 0.5,
    'sleep_duration': (15,30)
}


@click.command()
def cli():
    fileConfig(LOG_CONFIG, disable_existing_loggers=False)
    kraken = Kraken(key_path=API_KEY)
    strategy = TakeProfitMomentumStrategy('ETH', kraken, **config)
    strategy.run()
