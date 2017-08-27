import os
from logging.config import dictConfig

import click
import yaml

from cryptotrading.exchanges import Backtest, Kraken, Poloniex
from cryptotrading.strategy import TakeProfitMomentumStrategy, MFIMomentumStrategy


KRAKEN_API_KEY = os.path.expanduser('~/.kraken_api_key')
POLONIEX_API_KEY = os.path.expanduser('~/.poloniex_api_key')
LOG_CONFIG = 'cryptotrading/logging_conf.yaml'


tpm_config = {
    'base_currency': 'ETH',
    'quote_currency': 'USDT',
    'unit': 0.02,
    'macd_threshold': 0.5,
    'ohlc_interval': 60,  # in minutes
    'target_profit': .0225,
    'stop_loss': 0.0075,
    'buffer_percent': 0.0025,
    'sleep_duration': 15*60  # in seconds
}


mfi_config = {
    'base_currency': 'ETH',
    'quote_currency': 'USDT',
    'unit': 0.02,
    'ohlc_interval': 30,
    'mfi': (14, 80, 20),
    'macd_slope_min': .1,
    'sleep_duration': 0
}


def configure_logging():
    with open(LOG_CONFIG, 'rt') as f:
        log_config = yaml.safe_load(f.read())
        dictConfig(log_config)


@click.group()
@click.option('--exchange', type=click.Choice(['backtest', 'kraken', 'poloniex']), default='poloniex')
@click.pass_context
def cli(ctx, exchange):
    configure_logging()

    if exchange =='backtest':
        exchange_adapter = Backtest(POLONIEX_API_KEY, 'ETH', 'USDT', start='6/20/2017', end='8/20/2017', interval=30)
    elif exchange == 'kraken':
        exchange_adapter = Kraken(key_path=KRAKEN_API_KEY)
    else:
        exchange_adapter = Poloniex(key_path=POLONIEX_API_KEY)

    ctx.obj = {'exchange': exchange_adapter}


@cli.command()
@click.pass_context
def mfimomentum(ctx):
    exchange = ctx.obj.get('exchange')
    strategy = MFIMomentumStrategy(exchange, **mfi_config)
    strategy.run()


@cli.command()
@click.pass_context
def tpmomentum(ctx):
    exchange = ctx.obj.get('exchange')
    strategy = TakeProfitMomentumStrategy(exchange, **tpm_config)
    strategy.run()
