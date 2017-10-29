import io
import functools
import logging
import time
from collections import deque, namedtuple
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import progressbar
import tensorflow as tf

from qrypto import settings
from qrypto.backtest import Backtest
from qrypto.data.datasets import CompositeQLearnDataset
from qrypto.models.rnn_multitask import RNNMultiTaskLearner
from qrypto.strategy.qlearn.experience_buffer import ExperienceBuffer


tf.logging.set_verbosity(tf.logging.ERROR)
log = logging.getLogger(__name__)


root_dir = Path().resolve()
experiments_dir = root_dir/'experiments'
experiments_dir.mkdir(exist_ok=True)
summaries_dir = experiments_dir/'summaries'
summaries_dir.mkdir(exist_ok=True)

data_dir = root_dir/'data'


MAX_BUFFER_SIZE = 100000


class LearnStrategy(object):

    def __init__(self,
                 model_class,
                 model_name: str,
                 exchange,
                 base_currency: str,
                 quote_currency: str,
                 unit: float,
                 ohlc_interval: int = 5,
                 sleep_duration: int = 5,
                 **kwargs) -> None:
        self.model_class = model_class
        self.model_name = model_name
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.unit = unit
        self.ohlc_interval = ohlc_interval
        self.exchange = exchange
        self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.data_dir = data_dir/base_currency.lower()

        self.addtl_currencies = settings.addtl_currencies
        indicators = settings.get_indicators(base_currency, self.addtl_currencies)
        # csv_data = settings.get_csv_data(self.data_dir/'blockchain')
        gkg_file = self.data_dir/'gkg'/'gkg_data.txt'
        self.data = CompositeQLearnDataset(base_currency, ohlc_interval, indicators, gkg_file=gkg_file)

    def update(self):
        new_data = self.exchange.get_ohlc(self.base_currency, self.quote_currency, interval=self.ohlc_interval)
        self.data.update(new_data)

    def train(self,
              start: str,
              end: str,
              epochs: int = 20,
              validation_percent: float = 0.2,
              batch_size: int = 8,
              n_batches: int = 500,
              trace_days: int = 7,
              random_seed: int = None,
              **kwargs):
        """

        :param start:
        :param end:
        :param epochs: number of full passes over the dataset, including evaluation
        :param validation_percent:
        :param batch_size:
        :param n_batches: number of batches per epoch
        :param trace_days:
        :param random_seed:
        """

        # TODO: save training params to file for later reference
        # TODO: support validation_start date (instead of percent)

        n_examples = self._initialize_training_data(start, end)
        trace_length = (trace_days * 24 * 60) // self.ohlc_interval

        self.n_inputs = self.data.n_state_factors
        self.random = np.random.RandomState(random_seed)

        nan_buffer = self.data.skip_nans()
        n_examples -= nan_buffer + 1
        initial_step = nan_buffer

        train_steps = int(n_examples * (1. - validation_percent))
        validation_steps = int(n_examples * validation_percent)

        tf.reset_default_graph()
        global_step = tf.Variable(0, name='global_step', trainable=False)

        if random_seed:
            tf.set_random_seed(random_seed)

        self.model = self.model_class(self.model_name, self.n_inputs, summaries_dir=summaries_dir, **kwargs)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.data.set_to(initial_step)

            print('\nPopulating training data...')
            training_data = self._populate_training_data(train_steps)

            for epoch in range(epochs):
                self.data.set_to(initial_step)

                print('\nEpoch {}'.format(epoch))
                print('Training...')
                prog_bar = progressbar.ProgressBar(term_width=80)
                train_losses = []

                # Train the network
                for i in prog_bar(range(n_batches)):
                    rnn_state = self.model.initial_rnn_state(batch_size)
                    samples = training_data.sample(batch_size, trace_length)
                    inputs, labels = map(np.array, zip(*samples))
                    loss = self.model.update(sess, inputs, labels, trace_length, rnn_state)
                    train_losses.append(loss)

                # Compute error over training set
                print('Evaluating training set...')
                self.data.set_to(initial_step)
                train_accuracy, _, _ = self._evaluate(sess, train_steps, compute_losses=False, place_orders=False)

                # Compute error over validation set
                print('Evaluating validation set...')
                self.data.set_to(initial_step + train_steps, reset_orders=False)
                start_price = self.data.last_price
                val_accuracy, val_losses, returns = self._evaluate(sess, validation_steps)

                # Compute outperformance of market return
                market_return, algorithm_return = self._calculate_performance(returns, start_price)
                outperformance = algorithm_return - market_return
                print('Market return: {:.2f}%'.format(100 * market_return))
                print('Algorithm return: {:.2f}%'.format(100 * algorithm_return))
                print('Outperformance: {:+.2f}%'.format(100 * (outperformance)))

                # Add Tensorboard summaries
                epoch_summary = tf.Summary()
                for phase, phase_losses in zip(('train', 'validate'), (train_losses, val_losses)):
                    for i, losses in enumerate(zip(*phase_losses)):
                        tag_name = 'epoch/{}/{}_loss'.format(phase, self.tasks[i])
                        epoch_summary.value.add(simple_value=np.average(losses), tag=tag_name)
                epoch_summary.value.add(simple_value=np.average(train_accuracy), tag='epoch/train/accuracy')
                epoch_summary.value.add(simple_value=np.average(val_accuracy), tag='epoch/validate/accuracy')
                epoch_summary.value.add(simple_value=algorithm_return, tag='epoch/validate/return')
                self.model.summary_writer.add_summary(epoch_summary, epoch)
                self.model.summary_writer.add_summary(self._get_epoch_chart(epoch), epoch)
                self.model.summary_writer.flush()

    def _initialize_training_data(self, start, end):
        base_currency_data = Backtest(self.exchange, self.base_currency, self.quote_currency,
                                      start=start, end=end, interval=self.ohlc_interval,
                                      save_to_csv=self.data_dir/'market').all()
        self.data.init_data(base_currency_data, self.base_currency)

        for currency in self.addtl_currencies:
            currency_data = Backtest(self.exchange, currency, self.quote_currency, start=start, end=end,
                            interval=self.ohlc_interval).all()
            self.data.init_data(currency_data, currency)

        return len(base_currency_data)

    def _populate_training_data(self, n_steps: int):
        training_data = ExperienceBuffer(MAX_BUFFER_SIZE, self.random)

        for _ in range(n_steps):
            state = self.data.state()
            self.data.next()
            label = self._create_label(self.data)
            training_data.add((state, label))

        return training_data

    def _evaluate(self, session, n_steps, compute_losses: bool = True, place_orders: bool = True):
        prog_bar = progressbar.ProgressBar(term_width=80)
        initial_rnn_state = rnn_state = self.model.initial_rnn_state()

        returns = []
        accuracies = []
        val_losses = []

        for _ in prog_bar(range(n_steps)):
            state = self.data.state()
            prediction, new_rnn_state = self.model.predict(session, np.expand_dims(state, 0), 1, rnn_state, training=False)
            action_idx = self._order_strategy(prediction)

            _, cum_return = self.data.validate(action_idx, place_orders=place_orders)
            label = self._create_label(self.data)
            returns.append(cum_return)
            accuracies.append(action_idx == self._order_strategy(label, is_label=True))

            if compute_losses:
                loss = self.model.compute_loss(session, state, label, rnn_state)
                val_losses.append(loss)

            rnn_state = new_rnn_state

        return accuracies, val_losses, returns

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

    @staticmethod
    def _create_label(data):
        raise NotImplementedError

    @staticmethod
    def _order_strategy(prediction, is_label: bool = False):
        """Given a model prediction or a label, return the index of the action to be
        taken. E.g, return 0 for short, 1 for long.
        """
        raise NotImplementedError

    def run(self):
        pass
