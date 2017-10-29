import time

import numpy as np
import tensorflow as tf

from qrypto.models.utils import reduce_std


EPSILON = 10e-6
INITIAL_LOSS_PARAMS = [2., 2.]


class RNNMultiTaskLearner(object):

    def __init__(self,
                 scope: str,
                 n_inputs: int,
                 hidden_units: int = None,
                 learn_rate: float = 0.0005,
                 reg_strength: float = 0.1,
                 renorm_decay: float = 0.99,
                 dropout_prob: float = 0.,
                 rnn_dropout_prob: float = 0.,
                 rnn_layers: int = 1,
                 initial_vol_sigma: float = 2,
                 summaries_dir: str = None):
        self.scope = scope
        self.n_inputs = n_inputs
        self.n_hiddens = hidden_units or n_inputs
        self.rnn_layers = rnn_layers

        self.vol_outputs = np.array([])
        self.loss_params = {}

        # TODO: try using dense sparse dense regularization

        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, n_inputs], dtype=tf.float32, name='inputs')
            self.labels = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='labels')
            self.phase = tf.placeholder(dtype=tf.bool, name='phase')
            self.trace_length = tf.placeholder(dtype=tf.int32, name='trace_length')

            volatility_labels = self.labels[:,0]
            direction_labels = tf.to_int32(self.labels[:,1])

            batch_size = tf.reshape(tf.shape(self.inputs)[0] // self.trace_length, shape=[])

            norm_layer = tf.contrib.layers.batch_norm(self.inputs, scale=True, renorm=True, renorm_decay=renorm_decay, is_training=self.phase)
            norm_flat = tf.reshape(norm_layer, shape=[batch_size, self.trace_length, n_inputs])

            rnn_cell = tf.contrib.rnn.LSTMCell(num_units=n_inputs, state_is_tuple=True, activation=tf.nn.softsign, use_peepholes=True)
            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=1-rnn_dropout_prob)
            rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell] * rnn_layers, state_is_tuple=True)

            self.rnn_in = rnn_cell.zero_state(batch_size, dtype=tf.float32)
            rnn, self.rnn_state = tf.nn.dynamic_rnn(rnn_cell, norm_flat, dtype=tf.float32, initial_state=self.rnn_in)
            rnn = tf.reshape(rnn, shape=tf.shape(norm_layer))

            l1_reg = tf.contrib.layers.l1_regularizer(reg_strength)
            hidden_layer = tf.contrib.layers.fully_connected(rnn, self.n_hiddens, activation_fn=tf.nn.tanh)

            # Task 1: Estimate Volatility
            volatility_hidden = tf.contrib.layers.fully_connected(hidden_layer, self.n_hiddens, activation_fn=tf.nn.tanh, weights_regularizer=l1_reg)
            volatility_dropout = tf.layers.dropout(volatility_hidden, dropout_prob, training=self.phase)
            self.volatility_out = tf.contrib.layers.fully_connected(volatility_dropout, 1, activation_fn=None, weights_regularizer=l1_reg)
            self.volatility_out = tf.reshape(self.volatility_out, shape=[tf.shape(self.inputs)[0]])
            self.volatility_loss = tf.losses.mean_squared_error(volatility_labels, self.volatility_out)

            # Task 2: Classify Direction
            direction_hidden = tf.contrib.layers.fully_connected(hidden_layer, self.n_hiddens, activation_fn=tf.nn.tanh, weights_regularizer=l1_reg)
            direction_dropout = tf.layers.dropout(direction_hidden, dropout_prob, training=self.phase)
            self.direction_out = tf.contrib.layers.fully_connected(direction_dropout, 2, activation_fn=None, weights_regularizer=l1_reg)
            self.direction_out = tf.reshape(self.direction_out, shape=[tf.shape(self.inputs)[0], 2])
            direction_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=direction_labels, logits=self.direction_out)
            self.direction_loss = tf.reduce_mean(direction_losses)

            self.joint_loss = self._uncertainty_loss([self.volatility_loss, self.direction_loss])
            optimizer = tf.train.AdamOptimizer(learn_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.joint_loss, global_step=tf.contrib.framework.get_global_step())

            self.summaries = tf.summary.merge([
                tf.summary.histogram('normed_inputs', norm_layer),
                tf.summary.scalar('volatility_loss', self.volatility_loss),
                tf.summary.scalar('direction_loss', self.direction_loss),
                tf.summary.histogram('direction_loss_hist', direction_losses),
                tf.summary.histogram('volatility_predictions', self.volatility_out),
                tf.summary.histogram('direction_predictions', self.direction_out)
            ])

            self.summary_writer = None
            if summaries_dir:
                summary_dir = summaries_dir/'{}_{}'.format(scope, time.strftime('%Y%m%d_%H%M%S'))
                summary_dir.mkdir(exist_ok=True)
                self.summary_writer = tf.summary.FileWriter(str(summary_dir))

    def _uncertainty_loss(self, loss_ops, inital_values=INITIAL_LOSS_PARAMS):
        scaled_losses = []

        for i, loss in enumerate(loss_ops):
            loss_param = tf.Variable(inital_values[i], dtype=tf.float32, trainable=False, name='loss_param_{}'.format(i))
            self.loss_params[i] = loss_param
            scaled_loss_fn = ((1 / ((2 * loss_param) + EPSILON)) * loss) + tf.log(loss_param)
            scaled_losses.append(scaled_loss_fn)

        return tf.reduce_sum(scaled_losses)

    def predict(self, sess, state, trace_length, rnn_state):
        feed_dict = {
            self.inputs: state,
            self.phase: False,
            self.trace_length: trace_length,
            self.rnn_in: rnn_state
        }
        v_out, d_out, rnn = sess.run([self.volatility_out, self.direction_out, self.rnn_state], feed_dict)
        return (v_out, d_out), rnn

    def update(self, sess, state, labels, trace_length, rnn_state):
        feed_dict = {
            self.inputs: state,
            self.labels: labels,
            self.phase: True,
            self.trace_length: trace_length,
            self.rnn_in: rnn_state
        }

        tensors = [
            self.summaries,
            tf.contrib.framework.get_global_step(),
            self.train_op,
            self.volatility_loss,
            self.direction_loss,
            self.volatility_out
        ]

        summaries, step, _, v_loss, d_loss, v_out = sess.run(tensors, feed_dict)

        # Update loss parameters
        self.vol_outputs = np.append(self.vol_outputs, v_out)
        self.loss_params[0].assign(self.vol_outputs.var())

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, step)

        return (v_loss, d_loss)

    def compute_loss(self, sess, state, label, rnn_state):
        feed_dict = {
            self.inputs: np.array([state]),
            self.labels: np.array([label]),
            self.phase: False,
            self.trace_length: 1,
            self.rnn_in: rnn_state
        }
        v_loss, d_loss = sess.run([self.volatility_loss, self.direction_loss], feed_dict)
        return (v_loss, d_loss)

    def initial_rnn_state(self, size: int = 1):
        return [(np.zeros([size, self.n_inputs]), np.zeros([size, self.n_inputs]))] * self.rnn_layers
