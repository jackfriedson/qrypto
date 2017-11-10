import time

import numpy as np
import tensorflow as tf


EPSILON = 10e-9


class RegressorVarianceModel(object):

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
        self.n_inputs = n_inputs
        self.rnn_layers = rnn_layers
        self.scope = scope

        # TODO: try using dense sparse dense regularization

        with tf.variable_scope(scope):
            # Placeholders
            self.inputs = tf.placeholder(shape=[None, n_inputs], dtype=tf.float32, name='inputs')
            self.labels = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='labels')
            self.phase = tf.placeholder(dtype=tf.bool, name='phase')
            self.trace_length = tf.placeholder(dtype=tf.int32, name='trace_length')
            self.return_labels = self.labels[:,0]
            batch_size = tf.reshape(tf.shape(self.inputs)[0] // self.trace_length, shape=[])
            l1_reg = tf.contrib.layers.l1_regularizer(reg_strength)

            # Normalization Layer
            self.norm_layer = tf.contrib.layers.batch_norm(self.inputs, scale=True, renorm=True, renorm_decay=renorm_decay, is_training=self.phase)
            self.norm_flat = tf.reshape(self.norm_layer, shape=[batch_size, self.trace_length, n_inputs])

            # RNN
            rnn_cell = tf.contrib.rnn.LSTMCell(num_units=n_inputs, state_is_tuple=True, activation=tf.nn.softsign, use_peepholes=True)
            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=1-rnn_dropout_prob)
            rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell] * rnn_layers, state_is_tuple=True)
            self.rnn_in = rnn_cell.zero_state(batch_size, dtype=tf.float32)
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(rnn_cell, self.norm_flat, dtype=tf.float32, initial_state=self.rnn_in)
            self.rnn = tf.reshape(self.rnn, shape=tf.shape(self.norm_layer))

            # Hidden Layer
            n_hiddens = hidden_units or n_inputs
            self.hidden_layer = tf.contrib.layers.fully_connected(self.rnn, n_hiddens, activation_fn=tf.nn.tanh, weights_regularizer=l1_reg)
            self.dropout_layer = tf.layers.dropout(self.hidden_layer, dropout_prob, training=self.phase)

            # Output Layer
            self.return_out = tf.contrib.layers.fully_connected(self.dropout_layer, 1, activation_fn=None, weights_regularizer=l1_reg)
            self.return_out = tf.reshape(self.return_out, shape=[tf.shape(self.inputs)[0]])
            self.variance_out = tf.contrib.layers.fully_connected(self.dropout_layer, 1, activation_fn=None, weights_regularizer=l1_reg)
            self.variance_out = tf.square(self.variance_out) + EPSILON
            self.variance_out = tf.reshape(self.variance_out, shape=[tf.shape(self.inputs)[0]])

            # Losses
            self.return_losses = tf.losses.absolute_difference(self.return_labels, self.return_out, reduction='none')
            self.return_loss = tf.reduce_mean(self.return_losses)
            self.joint_losses = (self.return_loss / (2. * self.variance_out)) + (tf.log(self.variance_out) / 2.)
            self.joint_loss = tf.reduce_mean(self.joint_losses)

            self.outputs = [self.return_out, self.variance_out]
            self.losses = [self.joint_loss]

            self.optimizer = tf.train.AdamOptimizer(learn_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.joint_loss, global_step=tf.contrib.framework.get_global_step())

            # Summaries
            self.summaries = tf.summary.merge([
                tf.summary.histogram('normed_inputs', self.norm_layer),
                tf.summary.histogram('return_predictions', self.return_out),
                tf.summary.histogram('aleatoric_variance', self.variance_out),
                tf.summary.histogram('return_losses', self.return_losses),
                tf.summary.histogram('joint_losses', self.joint_losses),
                tf.summary.scalar('return_loss', self.return_loss),
                tf.summary.scalar('joint_loss', self.joint_loss)
            ])

            self.summary_writer = None
            if summaries_dir:
                summary_dir = summaries_dir/'{}_{}'.format(scope, time.strftime('%Y%m%d_%H%M%S'))
                summary_dir.mkdir(exist_ok=True)
                self.summary_writer = tf.summary.FileWriter(str(summary_dir))

    def predict(self, sess, state, trace_length, rnn_state, training: bool = True):
        feed_dict = {
            self.inputs: state,
            self.phase: training,
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

        summaries, global_step, _, losses = sess.run([self.summaries, tf.contrib.framework.get_global_step(),
                                                      self.train_op, self.losses], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)

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
