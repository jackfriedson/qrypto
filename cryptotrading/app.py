import os
from logging.config import dictConfig

import click
import yaml

from cryptotrading.exchanges import Kraken
from cryptotrading.strategy.momentum import TakeProfitMomentumStrategy


API_KEY = os.path.expanduser('~/.kraken_api_key')
LOG_CONFIG = 'cryptotrading/logging_conf.yaml'


config = {
    'unit': 0.02,
    'macd_threshold': 0.3,
    'target_profit': .0225,
    'stop_loss': 0.0075,
    'buffer_percent': 0.0025,
    'sleep_duration': (15, 30)
}


@click.command()
def cli():
    configure_logging()
    kraken = Kraken(key_path=API_KEY)
    strategy = TakeProfitMomentumStrategy('ETH', kraken, **config)
    strategy.run()

def configure_logging():
    with open(LOG_CONFIG, 'rt') as f:
        log_config = yaml.safe_load(f.read())
        dictConfig(log_config)
