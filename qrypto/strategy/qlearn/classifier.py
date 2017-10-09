import io
import functools
import logging
import time
from collections import namedtuple
from pathlib import Path
from typing import Tuple

import numpy as np
import progressbar
import tensorflow as tf

from qrypto.backtest import Backtest
from qrypto.data.datasets import QLearnDataset
from qrypto.data.indicators import BasicIndicator, DifferenceIndicator
from qrypto.strategy.qlearn.experience_buffer import ExperienceBuffer
from qrypto.strategy.qlearn.rnn_classifier import RNNClassifier


log = logging.getLogger(__name__)


root_dir = Path().resolve()
experiments_dir = root_dir/'experiments'
experiments_dir.mkdir(exist_ok=True)

summaries_dir = experiments_dir/'summaries'
summaries_dir.mkdir(exist_ok=True)

models_dir = experiments_dir/'models'
models_dir.mkdir(exist_ok=True)


class ClassifierStrategy(object):

    def __init__(self,
                 exchange,
                 base_currency: str,
                 quote_currency: str,
                 unit: float,
                 ohlc_interval: int = 5,
                 sleep_duration: int = 5,
                 confidence_thresholds: Tuple[float, float] = (0.5, 0.5),
                 **kwargs) -> None:
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.unit = unit
        self.ohlc_interval = ohlc_interval
        self.confidence_thresholds = confidence_thresholds
        self.exchange = exchange

        self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.models_dir = models_dir/self.timestamp

        indicators = [
            BasicIndicator('ppo'),
            BasicIndicator('adx')
        ]
        self.data = QLearnDataset(indicators=indicators, **kwargs)

    def update(self):
        new_data = self.exchange.get_ohlc(self.base_currency, self.quote_currency, interval=self.ohlc_interval)
        self.data.update(new_data)

    def train(self,
              start: str,
              end: str,
              n_slices: int = 10,
              n_epochs: int = 1,
              validation_percent: float = 0.2,
              gamma: float = 0.9,
              epsilon_start: float = 1.,
              epsilon_end: float = 0.,
              epsilon_decay: float = 2,
              replay_memory_start_size: int = 1000,
              replay_memory_max_size: int = 100000,
              batch_size: int = 8,
              trace_length: int = 16,
              update_target_every: int = 500,
              random_seed: int = None,
              load_model: str = None,
              **kwargs):

        # Initialize training data
        exchange_train = Backtest(self.exchange, self.base_currency, self.quote_currency,
                                  start=start, end=end, interval=self.ohlc_interval)
        self.data.init_data(exchange_train.all())
        n_inputs = self.data.n_state_factors
        n_outputs = 2
        random = np.random.RandomState(random_seed)

        # TODO: save training params to file for later reference

        total_steps = len(exchange_train.date_range)
        nan_buffer = self.data.start_training()
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

        cell = tf.contrib.rnn.LSTMCell(num_units=n_inputs, state_is_tuple=True, activation=tf.nn.softsign)
        classifier = RNNClassifier('rnn_classifier', cell, n_inputs, n_outputs, summaries_dir=summaries_dir, **kwargs)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for data_slice in range(n_slices):
                self.data.start_training(initial_step)

                print('Populating data...')
                replay_memory = ExperienceBuffer(replay_memory_max_size, random)
                for _ in range(train_steps):
                    price = self.data.last
                    state = self.data.state()
                    self.data.next()
                    label = 1 if self.data.last > price else 0
                    replay_memory.add((state, label))

                for epoch in range(n_epochs):
                    self.data.start_training(initial_step)
                    absolute_epoch = (data_slice * n_epochs) + epoch

                    print('\nSlice {}; Epoch {}'.format(data_slice, epoch))
                    print('Training...')
                    train_bar = progressbar.ProgressBar(term_width=80)
                    losses = []

                    n_batches = train_steps // batch_size // trace_length
                    # Train the network
                    for i in train_bar(range(10 * n_batches)):
                        rnn_state = (np.zeros([batch_size, n_inputs]), np.zeros([batch_size, n_inputs]))
                        samples = replay_memory.sample(batch_size, trace_length)
                        inputs, labels = map(np.array, zip(*samples))
                        loss = classifier.update(sess, inputs, labels, trace_length, rnn_state)
                        losses.append(loss)

                    saver.save(sess, str(self.models_dir/'model.ckpt'))

                    # Evaluate the model
                    print('Evaluating...')
                    self.data.start_training(initial_step + train_steps)
                    returns = []
                    confidences = []
                    predictions = []
                    start_price = self.data.last

                    rnn_state = (np.zeros([1, n_inputs]), np.zeros([1, n_inputs]))

                    for _ in range(validation_steps):
                        price = self.data.last
                        state = self.data.state()
                        _, probabilities, rnn_state = classifier.predict(sess, np.expand_dims(state, 0), 1, rnn_state, training=False)
                        prediction = np.argmax(probabilities)
                        confidence = probabilities[0][prediction]
                        _, cum_return = self.data.step_val(prediction)
                        label = 1 if self.data.last > price else 0

                        returns.append(cum_return)
                        confidences.append(confidence)
                        predictions.append(prediction == label)

                    # Compute outperformance of market return
                    market_return = (self.data.last / start_price) - 1.
                    position_value = start_price
                    for return_val in returns:
                        position_value *= 1 + return_val
                    algorithm_return = (position_value / start_price) - 1.
                    outperformance = algorithm_return - market_return
                    print('Market return: {:.2f}%'.format(100 * market_return))
                    print('Outperformance: {:+.2f}%'.format(100 * outperformance))

                    buf = io.BytesIO()
                    self.data.plot(save_to=buf)
                    buf.seek(0)
                    image = tf.image.decode_png(buf.getvalue(), channels=4)
                    image = tf.expand_dims(image, 0)
                    epoch_chart = tf.summary.image('epoch_{}'.format(absolute_epoch), image, max_outputs=1).eval()

                    # Add Tensorboard summaries
                    epoch_summary = tf.Summary()
                    epoch_summary.value.add(simple_value=np.average(losses), tag='epoch/train/averge_loss')
                    epoch_summary.value.add(simple_value=outperformance, tag='epoch/validate/outperformance')
                    epoch_summary.value.add(simple_value=np.average(confidences), tag='epoch/validate/average_confidence')
                    epoch_summary.value.add(simple_value=np.average(predictions), tag='epoch/validate/accuracy')
                    classifier.summary_writer.add_summary(epoch_summary, absolute_epoch)
                    classifier.summary_writer.add_summary(epoch_chart, absolute_epoch)
                    classifier.summary_writer.flush()

                # After all repeats, move to the next timeframe
                initial_step += validation_steps

    def run(self):
        pass
