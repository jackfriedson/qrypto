import os
from logging.config import dictConfig

import click
import yaml

from cryptotrading import settings
from cryptotrading.backtest import Backtest
from cryptotrading.exchanges import Kraken, Poloniex
from cryptotrading.strategy import TakeProfitMomentumStrategy, MFIMomentumStrategy, QTableStrategy, QNetworkStrategy

KRAKEN_API_KEY = os.path.expanduser('~/.kraken_api_key')
POLONIEX_API_KEY = os.path.expanduser('~/.poloniex_api_key')
LOG_CONFIG = 'cryptotrading/logging_conf.yaml'


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
@click.pass_context
def qlearn(ctx):
    exchange = ctx.obj.get('exchange')
    config = settings.get_config('qlearn')
    strategy = QTableStrategy(exchange, **config)
    strategy.train()


@cli.command()
@click.pass_context
def qlearnnet(ctx):
    exchange = ctx.obj.get('exchange')
    config = settings.get_config('qlearn')
    strategy = QNetworkStrategy(exchange, **config)
    strategy.train()


@cli.command()
@click.pass_context
def mfimomentum(ctx):
    exchange = ctx.obj.get('exchange')
    config = settings.get_config('mfi')
    ctx.obj['strategy'] = MFIMomentumStrategy(exchange, **config)


@cli.command()
@click.pass_context
def tpmomentum(ctx):
    exchange = ctx.obj.get('exchange')
    config = settings.get_config('tpm')
    ctx.obj['strategy'] = TakeProfitMomentumStrategy(exchange, **config)


@cli.command()
@click.pass_context
def run(ctx):
    ctx.obj['strategy'].run()


@cli.command()
@click.option('--base', type=str, default='BTC')
@click.option('--quote', type=str, default='USDT')
@click.option('--start', type=str, default='6/1/2017')
@click.option('--end', type=str, default='7/1/2017')
@click.option('--interval', type=int, default=5)
@click.pass_context
def test(ctx, base, quote, start, end, interval):
    exchange = ctx.obj.pop('exchange')
    test_exchange = Backtest(exchange, base, quote, start, end, interval)
    strategy = ctx.obj.pop('strategy')
    strategy.test(test_exchange)