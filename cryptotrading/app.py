import os
from logging.config import dictConfig

import click
import yaml

from cryptotrading.exchanges import Kraken
from cryptotrading.strategy import TakeProfitMomentumStrategy, MFIMomentumStrategy


API_KEY = os.path.expanduser('~/.kraken_api_key')
LOG_CONFIG = 'cryptotrading/logging_conf.yaml'


tpm_config = {
    'unit': 0.02,
    'macd_threshold': 0.5,
    'ohlc_interval': 60,  # in minutes
    'target_profit': .0225,
    'stop_loss': 0.0075,
    'buffer_percent': 0.0025,
    'sleep_duration': 15*60  # in seconds
}


mfi_config = {
    'unit': 0.02,
    'ohlc_interval': 60,
    'mfi': (14, 70, 30),
    'sleep_duration': 15*60
}


def configure_logging():
    with open(LOG_CONFIG, 'rt') as f:
        log_config = yaml.safe_load(f.read())
        dictConfig(log_config)


@click.group()
@click.pass_context
def cli(ctx):
    configure_logging()
    ctx.obj = {'exchange': Kraken(key_path=API_KEY)}


@cli.command()
@click.pass_context
def mfimomentum(ctx):
    exchange = ctx.obj.get('exchange')
    strategy = MFIMomentumStrategy('ETH', exchange, **mfi_config)
    strategy.run()


@cli.command()
@click.pass_context
def simplemomentum(ctx):
    exchange = ctx.obj.get('exchange')
    strategy = TakeProfitMomentumStrategy('ETH', exchange, **tpm_config)
    strategy.run()
