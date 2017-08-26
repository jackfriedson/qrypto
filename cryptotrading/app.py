import os
from logging.config import dictConfig

import click
import yaml

from cryptotrading.exchanges import Kraken, Poloniex
from cryptotrading.strategy import TakeProfitMomentumStrategy, MFIMomentumStrategy


KRAKEN_API_KEY = os.path.expanduser('~/.kraken_api_key')
POLONIEX_API_KEY = os.path.expanduser('~/.poloniex_api_key')
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
    'unit': 0.001,
    'ohlc_interval': 120,
    'mfi': (14, 70, 30),
    'sleep_duration': 5*60
}


def configure_logging():
    with open(LOG_CONFIG, 'rt') as f:
        log_config = yaml.safe_load(f.read())
        dictConfig(log_config)


@click.group()
@click.option('--exchange', type=click.Choice(['kraken, poloniex']), default='poloniex')
@click.pass_context
def cli(ctx, exchange):
    configure_logging()

    if exchange == 'kraken':
        exchange_adapter = Kraken(key_path=KRAKEN_API_KEY)
    else:
        exchange_adapter = Poloniex(key_path=POLONIEX_API_KEY)

    ctx.obj = {'exchange': exchange_adapter}


@cli.command()
@click.pass_context
def mfimomentum(ctx):
    exchange = ctx.obj.get('exchange')
    strategy = MFIMomentumStrategy('BTC', exchange, quote_currency='USDT', **mfi_config)
    strategy.run()


@cli.command()
@click.pass_context
def simplemomentum(ctx):
    exchange = ctx.obj.get('exchange')
    strategy = TakeProfitMomentumStrategy('BTC', exchange, **tpm_config)
    strategy.run()
