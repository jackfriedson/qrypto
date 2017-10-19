import os
from logging.config import dictConfig
from pathlib import Path

import click
import matplotlib
matplotlib.use('Agg')  # Force matplotlib to not use x-windows backend
import yaml

from qrypto import settings
from qrypto.backtest import Backtest
from qrypto.exchanges import Kraken, Poloniex
from qrypto.strategy import (TakeProfitMomentumStrategy, MFIMomentumStrategy, QTableStrategy, QNetworkStrategy,
                             ClassifierStrategy, RegressorStrategy)


RANDOM_SEED = 12345

KRAKEN_API_KEY = os.path.expanduser('~/.kraken_api_key')
POLONIEX_API_KEY = os.path.expanduser('~/.poloniex_api_key')
LOG_CONFIG = 'qrypto/logging_conf.yaml'


log_dir = Path().resolve()/'logs'
log_dir.mkdir(exist_ok=True)


def configure_logging():
    with open(LOG_CONFIG, 'rt') as f:
        log_config = yaml.safe_load(f.read())
        log_config['handlers']['file']['filename'] = str(log_dir/'all.log')
        log_config['handlers']['order']['filename'] = str(log_dir/'orders.log')
        dictConfig(log_config)


@click.group(chain=True)
@click.option('--exchange', type=click.Choice(['kraken', 'poloniex']), default='poloniex')
@click.pass_context
def cli(ctx, exchange):
    configure_logging()

    if exchange == 'kraken':
        exchange_adapter = Kraken(key_path=KRAKEN_API_KEY)
    elif exchange == 'poloniex':
        exchange_adapter = Poloniex(key_path=POLONIEX_API_KEY)
    # elif exchange == 'cryptowatch':
    #     exchange_adapter = Cryptowatch()

    ctx.obj = {'exchange': exchange_adapter}


@cli.command()
@click.pass_context
def qlearn(ctx):
    exchange = ctx.obj.get('exchange')
    config = settings.get_config('qlearn')
    strategy = QTableStrategy(exchange, **config)
    strategy.train()


@cli.command()
@click.option('--train-start', type=str, default='6/1/2017')
@click.option('--train-end', type=str, default='10/1/2017')
@click.option('--n-slices', type=int, default=10)
@click.option('--n-epochs', type=int, default=1)
@click.option('--validation-percent', type=float, default=0.2)
@click.option('--learn-rate', type=float, default=0.001)
@click.option('--gamma', type=float, default=0.9)
@click.pass_context
def qlearnnet(ctx, train_start, train_end, **kwargs):
    exchange = ctx.obj.get('exchange')
    config = settings.get_config('qlearn')
    strategy = QNetworkStrategy(exchange, **config)
    strategy.train(train_start, train_end, random_seed=RANDOM_SEED, **kwargs)


@cli.command()
@click.option('--train-start', type=str, default='6/1/2017')
@click.option('--train-end', type=str, default='10/11/2017')
@click.option('--n-slices', type=int, default=20)
@click.option('--n-epochs', type=int, default=1)
@click.option('--validation-percent', type=float, default=0.05)
@click.option('--softmax-threshold', type=float, default=0.5)
@click.option('--target-period', type=int, default=1)
@click.option('--learn-rate', type=float, default=0.005)
@click.option('--hidden-units', type=int, default=None)
@click.option('--batch-size', type=int, default=16)
@click.option('--batch-repeats', type=int, default=40)
@click.option('--dropout-keep-prob', type=float, default=1.0)
@click.option('--trace-length', type=int, default=32)
@click.option('--rnn-layers', type=int, default=1)
@click.pass_context
def classifier(ctx, train_start, train_end, **kwargs):
    exchange = ctx.obj.get('exchange')
    config = settings.get_config('qlearn')
    strategy = ClassifierStrategy(exchange, **config)
    strategy.train(train_start, train_end, random_seed=RANDOM_SEED, **kwargs)

@cli.command()
@click.option('--train-start', type=str, default='6/1/2017')
@click.option('--train-end', type=str, default='10/11/2017')
@click.option('--n-slices', type=int, default=20)
@click.option('--n-epochs', type=int, default=1)
@click.option('--validation-percent', type=float, default=0.05)
@click.option('--softmax-threshold', type=float, default=0.5)
@click.option('--target-period', type=int, default=1)
@click.option('--learn-rate', type=float, default=0.005)
@click.option('--hidden-units', type=int, default=None)
@click.option('--batch-size', type=int, default=16)
@click.option('--batch-repeats', type=int, default=80)
@click.option('--dropout-keep-prob', type=float, default=1.0)
@click.option('--trace-length', type=int, default=32)
@click.option('--rnn-layers', type=int, default=2)
@click.pass_context
def regressor(ctx, train_start, train_end, **kwargs):
    exchange = ctx.obj.get('exchange')
    config = settings.get_config('qlearn')
    strategy = RegressorStrategy(exchange, **config)
    strategy.train(train_start, train_end, random_seed=RANDOM_SEED, **kwargs)

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
