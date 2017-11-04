from pathlib import Path

import numpy as np
import progressbar
import tensorflow as tf

from qrypto import settings
from qrypto.backtest import Backtest
from qrypto.data.datasets import QLearnDataset
from qrypto.models import FeatureLearningModel
from qrypto.strategy import LearnStrategy


root_dir = Path().resolve()
experiments_dir = root_dir/'experiments'
experiments_dir.mkdir(exist_ok=True)
summaries_dir = experiments_dir/'summaries'
summaries_dir.mkdir(exist_ok=True)

data_dir = root_dir/'data'


MIN_OHLC_INTERVAL = 5


class DSDStrategy(LearnStrategy):
    tasks = ['return']

    def __init__(self, *args, **kwargs):
        super(DSDStrategy, self).__init__(FeatureLearningModel, 'dsd_learner', *args, **kwargs)
        indicators = settings.get_indicators_full()
        self.addtl_currencies = []
        self.data = QLearnDataset(self.ohlc_interval, indicators=indicators)

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
        self._initialize_training_data(start, end)
        n_examples = self.data.n_examples
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
                for i, losses in enumerate(zip(*val_losses)):
                    tag_name = 'epoch/validate/{}_loss'.format(self.tasks[i])
                    epoch_summary.value.add(simple_value=np.average(losses), tag=tag_name)
                epoch_summary.value.add(simple_value=np.average(train_accuracy), tag='epoch/train/accuracy')
                epoch_summary.value.add(simple_value=np.average(val_accuracy), tag='epoch/validate/accuracy')
                epoch_summary.value.add(simple_value=algorithm_return, tag='epoch/validate/return')
                self.model.summary_writer.add_summary(epoch_summary, epoch)
                self.model.summary_writer.add_summary(self._get_epoch_chart(epoch), epoch)
                self.model.summary_writer.flush()

    def _initialize_training_data(self, start, end):
        base_currency_data = Backtest(self.exchange, self.base_currency, self.quote_currency,
                                      start=start, end=end, interval=MIN_OHLC_INTERVAL,
                                      save_to_csv=self.data_dir/'market').all()
        self.data.init_data(base_currency_data)

    @staticmethod
    def _create_label(data):
        return np.array([data.period_return])

    @staticmethod
    def _order_strategy(output, is_label: bool = False):
        return 1 if output[0] > 0 else 0
