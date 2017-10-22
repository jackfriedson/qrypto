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

from qrypto import settings
from qrypto.backtest import Backtest
from qrypto.data.datasets import CompositeQLearnDataset
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

csv_dir = root_dir/'csv_data'

addtl_currencies = ['ETH', 'LTC', 'ETC']


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
        self.csv_dir = csv_dir/base_currency.lower()

        indicators = settings.get_indicators(base_currency, addtl_currencies)
        csv_data = settings.get_csv_data(self.csv_dir/'blockchain')
        self.data = CompositeQLearnDataset(base_currency, ohlc_interval, indicators, csv_data)

    def update(self):
        new_data = self.exchange.get_ohlc(self.base_currency, self.quote_currency, interval=self.ohlc_interval)
        self.data.update(new_data)

    def train(self,
              start: str,
              end: str,
              n_slices: int = 10,
              slice_repeats: int = 1,
              validation_percent: float = 0.2,
              minimum_gain: float = 0.,
              max_buffer_size: int = 100000,
              batch_size: int = 8,
              train_iters: int = 2500,
              rnn_layers: int = 1,
              trace_days: int = 7,
              random_seed: int = None,
              **kwargs):
        # TODO: save training params to file for later reference

        n_datapoints = self._initialize_training_data(start, end)
        trace_length = (trace_days * 24 * 60) // self.ohlc_interval

        # TODO: consider moving these to init?
        self.n_inputs = self.data.n_state_factors
        self.n_outputs = 2
        self.rnn_layers = rnn_layers
        self.minimum_gain = minimum_gain
        self.random = np.random.RandomState(random_seed)
        self.max_buffer_size = max_buffer_size

        nan_buffer = self.data.skip_nans()
        n_datapoints -= nan_buffer + 1
        initial_step = nan_buffer

        step_ratio = 1. / (1. + ((n_slices - 1) * validation_percent))
        iter_steps = int(step_ratio * n_datapoints)
        train_steps = int(iter_steps * (1. - validation_percent))
        validation_steps = int(iter_steps * validation_percent)

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

                for repeat in range(slice_repeats):
                    self.data.set_to(initial_step)
                    iteration = (data_slice * slice_repeats) + repeat

                    print('\nSlice {}; Repeat {}'.format(data_slice, repeat))
                    print('Training...')
                    prog_bar = progressbar.ProgressBar(term_width=80)
                    losses = []

                    # Train the network
                    for i in prog_bar(range(train_iters)):
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
                    market_return, algorithm_return = self._calculate_performance(returns, start_price)
                    outperformance = algorithm_return - market_return
                    print('Market return: {:.2f}%'.format(100 * market_return))
                    print('Algorithm return: {:.2f}%'.format(100 * algorithm_return))
                    print('Outperformance: {:+.2f}%'.format(100 * (outperformance)))

                    # Add Tensorboard summaries
                    iteration_summary = tf.Summary()
                    iteration_summary.value.add(simple_value=np.average(losses), tag='epoch/train/loss')
                    iteration_summary.value.add(simple_value=np.average(train_error), tag='epoch/train/error')
                    iteration_summary.value.add(simple_value=np.average(train_accuracy), tag='epoch/train/accuracy')
                    iteration_summary.value.add(simple_value=np.average(val_error), tag='epoch/validate/error')
                    iteration_summary.value.add(simple_value=np.average(val_accuracy), tag='epoch/validate/accuracy')
                    iteration_summary.value.add(simple_value=algorithm_return, tag='epoch/validate/return')
                    regressor.summary_writer.add_summary(iteration_summary, iteration)
                    regressor.summary_writer.add_summary(self._get_epoch_chart(iteration), iteration)
                    regressor.summary_writer.flush()

                # After all repeats, move to the next timeframe
                initial_step += validation_steps

    def _initialize_training_data(self, start, end):
        base_currency_data = Backtest(self.exchange, self.base_currency, self.quote_currency,
                                      start=start, end=end, interval=self.ohlc_interval,
                                      save_to_csv=self.csv_dir/'market').all()
        self.data.init_data(base_currency_data, self.base_currency)

        for currency in addtl_currencies:
            currency_data = Backtest(self.exchange, currency, self.quote_currency, start=start, end=end,
                            interval=self.ohlc_interval).all()
            self.data.init_data(currency_data, currency)

        return len(base_currency_data)

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
            place_orders = place_orders and self._order_strategy(prediction, self.minimum_gain)
            _, cum_return = self.data.validate(action_idx, place_orders=place_orders)
            returns.append(cum_return)

            actual = (self.data.last_price / price) - 1.
            difference = abs(prediction - actual)
            differences.append(difference)
            correct_direction = (prediction >= 0 and actual >= 0) or (prediction < 0 and actual < 0)
            correct_directions.append(correct_direction)

        return differences, correct_directions, returns

    @staticmethod
    def _order_strategy(predicted_return, minimum_gain):
        return predicted_return > minimum_gain or predicted_return < 0

    def _calculate_performance(self, returns, start_price):
        market_return = (self.data.last_price / start_price) - 1.
        position_value = start_price
        for return_val in returns:
            position_value *= 1 + return_val
        algorithm_return = (position_value / start_price) - 1.
        return market_return, algorithm_return

    def _get_epoch_chart(self, epoch):
        buf = io.BytesIO()
        self.data.plot(save_to=buf)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return tf.summary.image('epoch_{}'.format(epoch), image, max_outputs=1).eval()

    def run(self):
        pass
