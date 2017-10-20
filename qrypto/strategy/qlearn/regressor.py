import io
import functools
import logging
import time
from collections import deque, namedtuple
from pathlib import Path
from typing import List, Tuple

import numpy as np
import progressbar
import tensorflow as tf

from qrypto.backtest import Backtest
from qrypto.data.datasets import CompositeQLearnDataset
from qrypto.data.indicators import BasicIndicator
from qrypto.models.rnn_regressor import RNNRegressor
from qrypto.strategy.qlearn.experience_buffer import ExperienceBuffer


tf.logging.set_verbosity(tf.logging.ERROR)
log = logging.getLogger(__name__)


root_dir = Path().resolve()
experiments_dir = root_dir/'experiments'
experiments_dir.mkdir(exist_ok=True)

summaries_dir = experiments_dir/'summaries'
summaries_dir.mkdir(exist_ok=True)

models_dir = experiments_dir/'models'
models_dir.mkdir(exist_ok=True)


class RegressorStrategy(object):

    def __init__(self,
                 exchange,
                 base_currency: str,
                 quote_currency: str,
                 unit: float,
                 ohlc_interval: int = 5,
                 sleep_duration: int = 5,
                 **kwargs) -> None:
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.unit = unit
        self.ohlc_interval = ohlc_interval
        self.exchange = exchange

        self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.models_dir = models_dir/self.timestamp

        configs = {
            base_currency: [
                BasicIndicator('rsi', 6),
                BasicIndicator('rsi', 12),
                BasicIndicator('mom', 1),
                BasicIndicator('mom', 3),
                BasicIndicator('obv'),
                BasicIndicator('adx', 14),
                BasicIndicator('adx', 20),
                BasicIndicator('macd'),
                BasicIndicator('bbands'),
                BasicIndicator('willr'),
                BasicIndicator('atr', 14),
                BasicIndicator('rocr', 3),
                BasicIndicator('rocr', 12),
                BasicIndicator('cci', 12),
                BasicIndicator('cci', 20),
                BasicIndicator('sma', 3),
                BasicIndicator('sma', 6),
                BasicIndicator('ema', 6),
                BasicIndicator('ema', 12),
                BasicIndicator('ema', 26),
                BasicIndicator('wma', 6),
                BasicIndicator('mfi', 14),
                BasicIndicator('trix'),
                BasicIndicator('stoch'),
                BasicIndicator('stochrsi'),
                BasicIndicator('ad'),
                BasicIndicator('adosc')
            ],
            'ETH': [
                BasicIndicator('mom', 1),
                BasicIndicator('mom', 6),
                BasicIndicator('mom', 12)
            ],
            'LTC': [
                BasicIndicator('mom', 1),
                BasicIndicator('mom', 6),
                BasicIndicator('mom', 12)
            ],
            'ETC': [
                BasicIndicator('mom', 1),
                BasicIndicator('mom', 6),
                BasicIndicator('mom', 12)
            ]
        }
        self.data = CompositeQLearnDataset(base_currency, configs)

    def update(self):
        new_data = self.exchange.get_ohlc(self.base_currency, self.quote_currency, interval=self.ohlc_interval)
        self.data.update(new_data)

    def _initialize_training_data(self, start, end, additional_currencies: List[str] = None):
        additional_currencies = additional_currencies or []

        # Initialize core currency data
        exchange_train = Backtest(self.exchange, self.base_currency, self.quote_currency,
                                  start=start, end=end, interval=self.ohlc_interval).all()
        self.data.init_data(exchange_train, self.base_currency)

        # Initialize additional currency data
        for currency in additional_currencies:
            currency_data = Backtest(self.exchange, currency, self.quote_currency, start=start, end=end,
                            interval=self.ohlc_interval).all()
            self.data.init_data(currency_data, currency)

        return len(exchange_train.date_range)

    def train(self,
              start: str,
              end: str,
              n_slices: int = 10,
              n_epochs: int = 1,
              validation_percent: float = 0.2,
              prediction_threshold: float = 0.,
              max_buffer_size: int = 100000,
              target_period: int = 1,
              batch_size: int = 8,
              batch_repeats: int = 10,
              rnn_layers: int = 1,
              trace_length: int = 16,
              random_seed: int = None,
              **kwargs):
        # TODO: save training params to file for later reference

        total_steps = self._initialize_training_data(start, end, ['ETH', 'LTC', 'ETC'])

        # TODO: move these to init
        self.n_inputs = self.data.n_state_factors
        self.n_outputs = 2
        self.rnn_layers = rnn_layers
        self.prediction_threshold = prediction_threshold
        self.random = np.random.RandomState(random_seed)
        self.max_buffer_size = max_buffer_size
        self.target_period = target_period

        nan_buffer = self.data.set_to()
        total_steps -= nan_buffer + 1
        initial_step = nan_buffer

        epoch_step_ratio = 1. / (1. + ((n_slices - 1) * validation_percent))
        epoch_steps = int(epoch_step_ratio * total_steps)
        train_steps = int(epoch_steps * (1. - validation_percent))
        validation_steps = int(epoch_steps * validation_percent)

        tf.reset_default_graph()
        global_step = tf.Variable(0, name='global_step', trainable=False)

        if random_seed:
            tf.set_random_seed(random_seed)

        regressor = RNNRegressor('rnn_regressor', self.n_inputs, self.n_outputs, rnn_layers=self.rnn_layers,
                                  summaries_dir=summaries_dir, **kwargs)

        # saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for data_slice in range(n_slices):
                self.data.set_to(initial_step)

                print('\nPopulating training data...')
                training_data = self._populate_training_data(train_steps)

                for epoch in range(n_epochs):
                    self.data.set_to(initial_step)
                    abs_epoch = (data_slice * n_epochs) + epoch

                    print('\nSlice {}; Epoch {}'.format(data_slice, epoch))
                    print('Training...')
                    prog_bar = progressbar.ProgressBar(term_width=80)
                    losses = []

                    n_batches = train_steps // batch_size // trace_length
                    # Train the network
                    for i in prog_bar(range(batch_repeats * n_batches)):
                        rnn_state = self._initial_rnn_state(batch_size)
                        samples = training_data.sample(batch_size, trace_length)
                        inputs, labels = map(np.array, zip(*samples))
                        loss = regressor.update(sess, inputs, labels, trace_length, rnn_state)
                        losses.append(loss)

                    # saver.save(sess, str(self.models_dir/'model.ckpt'))

                    # Compute error over training set
                    print('Evaluating training set...')
                    self.data.set_to(initial_step)
                    train_error, train_accuracy, _ = self._evaluate(sess, regressor, train_steps, place_orders=False)

                    # Compute error over validation set
                    print('Evaluating validation set...')
                    self.data.set_to(initial_step + train_steps, reset_orders=False)
                    start_price = self.data.last_price
                    val_error, val_accuracy, returns = self._evaluate(sess, regressor, validation_steps)

                    # Compute outperformance of market return
                    market_return, outperformance = self._calculate_performance(returns, start_price)
                    print('Market return: {:.2f}%'.format(100 * market_return))
                    print('Outperformance: {:+.2f}%'.format(100 * outperformance))

                    # Add Tensorboard summaries
                    epoch_summary = tf.Summary()
                    epoch_summary.value.add(simple_value=np.average(losses), tag='epoch/train/loss')
                    epoch_summary.value.add(simple_value=np.average(train_error), tag='epoch/train/error')
                    epoch_summary.value.add(simple_value=np.average(train_accuracy), tag='epoch/train/accuracy')
                    epoch_summary.value.add(simple_value=np.average(val_error), tag='epoch/validate/error')
                    epoch_summary.value.add(simple_value=np.average(val_accuracy), tag='epoch/validate/accuracy')
                    epoch_summary.value.add(simple_value=outperformance, tag='epoch/validate/outperformance')
                    regressor.summary_writer.add_summary(epoch_summary, abs_epoch)
                    regressor.summary_writer.add_summary(self._get_epoch_chart(abs_epoch), abs_epoch)
                    regressor.summary_writer.flush()

                # After all repeats, move to the next timeframe
                initial_step += validation_steps

    def _populate_training_data(self, n_steps: int):
        training_data = ExperienceBuffer(self.max_buffer_size, self.random)

        for _ in range(n_steps):
            price = self.data.last_price
            state = self.data.state()

            self.data.next()

            label = (self.data.last_price / price) - 1.
            training_data.add((state, label))

        return training_data

    def _initial_rnn_state(self, size: int = 1):
        return [(np.zeros([size, self.n_inputs]), np.zeros([size, self.n_inputs]))] * self.rnn_layers

    def _evaluate(self, session, regressor, n_steps, place_orders: bool = True):
        prog_bar = progressbar.ProgressBar(term_width=80)
        initial_rnn_state = rnn_state = self._initial_rnn_state()

        returns = []
        differences = []
        correct_directions = []

        for _ in prog_bar(range(n_steps)):
            price = self.data.last_price
            state = self.data.state()
            prediction, rnn_state = regressor.predict(session, np.expand_dims(state, 0), 1, rnn_state, training=False)
            prediction = prediction[0]

            action_idx = 1 if prediction > 0 else 0
            place_orders = place_orders and abs(prediction) > self.prediction_threshold
            _, cum_return = self.data.validate(action_idx, place_orders=place_orders)
            returns.append(cum_return)

            actual = (self.data.last_price / price) - 1.
            difference = abs(prediction - actual)
            differences.append(difference)
            correct_direction = (prediction >= 0 and actual >= 0) or (prediction < 0 and actual < 0)
            correct_directions.append(correct_direction)

        return differences, correct_directions, returns

    def _calculate_performance(self, returns, start_price):
        market_return = (self.data.last_price / start_price) - 1.
        position_value = start_price
        for return_val in returns:
            position_value *= 1 + return_val
        algorithm_return = (position_value / start_price) - 1.
        outperformance = algorithm_return - market_return
        return market_return, outperformance

    def _get_epoch_chart(self, epoch):
        buf = io.BytesIO()
        self.data.plot(save_to=buf)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return tf.summary.image('epoch_{}'.format(epoch), image, max_outputs=1).eval()

    def run(self):
        pass