import os
from logging.config import dictConfig

import click
import yaml

from cryptotrading.backtest import Backtest
from cryptotrading.exchanges import Kraken, Poloniex
from cryptotrading.strategy import TakeProfitMomentumStrategy, MFIMomentumStrategy, QTableStrategy

KRAKEN_API_KEY = os.path.expanduser('~/.kraken_api_key')
POLONIEX_API_KEY = os.path.expanduser('~/.poloniex_api_key')
LOG_CONFIG = 'cryptotrading/logging_conf.yaml'


tpm_config = {
    'base_currency': 'ETH',
    'quote_currency': 'USDT',
    'unit': 0.02,
    'macd_threshold': 0.3,
    'ohlc_interval': 30,  # in minutes
    'target_profit': .0225,
    'stop_loss': 0.0075,
    'sleep_duration': 0
    # 'sleep_duration': 15*60  # in seconds
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


qlearn_config = {
    'base_currency': 'ETH',
    'quote_currency': 'USDT',
    'unit': 1,
    'ohlc_interval': 5,
    # 'rsi': 14,
    # 'mfi': 14,
    'momentum': 12,
    'sleep_duration': 0
}


def configure_logging():
    with open(LOG_CONFIG, 'rt') as f:
        log_config = yaml.safe_load(f.read())
        dictConfig(log_config)


@click.group(chain=True)
@click.option('--exchange', type=click.Choice(['kraken', 'poloniex']), default='poloniex')
@click.pass_context
def cli(ctx, exchange):
    configure_logging()

    if exchange == 'kraken':
        exchange_adapter = Kraken(key_path=KRAKEN_API_KEY)
    else:
        exchange_adapter = Poloniex(key_path=POLONIEX_API_KEY)

    ctx.obj = {'exchange': exchange_adapter}


@cli.command()
@click.option('--base', type=str, default='BTC')
@click.option('--quote', type=str, default='USDT')
@click.option('--start', type=str, default='6/1/2017')
@click.option('--end', type=str, default='7/1/2017')
@click.option('--interval', type=int, default=5)
@click.pass_context
def backtest(ctx, base, quote, start, end, interval):
    exchange = ctx.obj.pop('exchange')
    ctx.obj['exchange'] = Backtest(exchange, base, quote, start, end, interval)

@cli.command()
@click.pass_context
def qlearn(ctx):
    exchange = ctx.obj.get('exchange')
    strategy = QTableStrategy(exchange, **qlearn_config)
    strategy.train()

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
