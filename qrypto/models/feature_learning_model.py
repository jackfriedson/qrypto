import time
from collections import deque

import numpy as np
import tensorflow as tf
from scipy.stats import entropy

from qrypto.models.utils import reduce_std


class FeatureLearningModel(object):

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
                 summaries_dir: str = None):
        self.scope = scope
        self.n_inputs = n_inputs
        self.n_hiddens = hidden_units or n_inputs
        self.rnn_layers = rnn_layers

        self.loss_params = {}

        # TODO: try using dense sparse dense regularization

        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, n_inputs], dtype=tf.float32, name='inputs')
            self.labels = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='labels')
            self.phase = tf.placeholder(dtype=tf.bool, name='phase')
            self.trace_length = tf.placeholder(dtype=tf.int32, name='trace_length')

            return_labels = self.labels[:,0]

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
            hidden_1 = tf.contrib.layers.fully_connected(rnn, self.n_hiddens, activation_fn=tf.nn.tanh, scope='hidden_layer')
            dropout_1 = tf.layers.dropout(hidden_1, dropout_prob, training=self.phase)

            # Task 1: Estimate Return
            return_hidden = tf.contrib.layers.fully_connected(dropout_1, self.n_hiddens, activation_fn=tf.nn.tanh, weights_regularizer=l1_reg)
            return_dropout = tf.layers.dropout(return_hidden, dropout_prob, training=self.phase)
            self.return_out = tf.contrib.layers.fully_connected(return_dropout, 1, activation_fn=None, weights_regularizer=l1_reg)
            self.return_out = tf.reshape(self.return_out, shape=[tf.shape(self.inputs)[0]])
            self.return_loss = tf.losses.mean_squared_error(return_labels, self.return_out)

            self.joint_loss = self.return_loss
            optimizer = tf.train.AdamOptimizer(learn_rate)

            self.outputs = [self.return_out]
            self.losses = [self.return_loss]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.joint_loss, global_step=tf.contrib.framework.get_global_step())

            self.summaries = [
                tf.summary.histogram('normed_inputs', norm_layer),
                tf.summary.histogram('return_predictions', self.return_out),
                tf.summary.scalar('return_loss', self.return_loss),
                tf.summary.scalar('joint_loss', self.joint_loss),
            ]
            # self.summaries.extend([tf.summary.scalar('loss_param_{}'.format(i), param) for i, param in self.loss_params.items()])
            self.summaries = tf.summary.merge(self.summaries)

            self.summary_writer = None
            if summaries_dir:
                summary_dir = summaries_dir/'{}_{}'.format(scope, time.strftime('%Y%m%d_%H%M%S'))
                summary_dir.mkdir(exist_ok=True)
                self.summary_writer = tf.summary.FileWriter(str(summary_dir))

    @staticmethod
    def get_tf_var(name):
        for var in tf.global_variables():
            if var.name.startswith(name):
                return var
        return None

    def prune_connections(self, sess, sparsity: float = 0.1):
        weights = self.get_tf_var('{}/hidden_layer/weights'.format(self.scope))
        weight_vals = weights.eval(sess)
        abs_weight_vals = np.abs(weight_vals)
        sorted_weights = np.sort(abs_weight_vals, axis=None)
        cutoff_idx = int(len(sorted_weights) * sparsity)
        cutoff_val = sorted_weights[cutoff_idx]
        mask = abs_weight_vals < cutoff_val
        weight_vals[mask] = 0.
        weights.assign(weight_vals)

    def predict(self, sess, state, trace_length, rnn_state):
        feed_dict = {
            self.inputs: state,
            self.phase: False,
            self.trace_length: trace_length,
            self.rnn_in: rnn_state
        }
        return sess.run([self.outputs, self.rnn_state], feed_dict)

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
            self.losses,
            self.outputs
        ]

        summaries, step, _, losses, outputs = sess.run(tensors, feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, step)

        return losses

    def compute_loss(self, sess, state, label, rnn_state):
        feed_dict = {
            self.inputs: np.array([state]),
            self.labels: np.array([label]),
            self.phase: False,
            self.trace_length: 1,
            self.rnn_in: rnn_state
        }
        return sess.run(self.losses, feed_dict)

    def initial_rnn_state(self, size: int = 1):
        return [(np.zeros([size, self.n_inputs]), np.zeros([size, self.n_inputs]))] * self.rnn_layers
