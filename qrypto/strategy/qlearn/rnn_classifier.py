import time

import numpy as np
import tensorflow as tf


class RNNClassifier(object):

    def __init__(self,
                 scope: str,
                 n_inputs: int,
                 n_outputs: int,
                 hidden_units: int = None,
                 learn_rate: float = 0.0005,
                 renorm_decay: float = 0.99,
                 dropout_keep_prob: float = 1.0,
                 rnn_layers: int = 1,
                 attention_depth: int = 1,
                 summaries_dir: str = None):
        self.scope = scope

        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, n_inputs], dtype=tf.float32, name='inputs')
            self.labels = tf.placeholder(shape=[None], dtype=tf.int32, name='labels')
            self.phase = tf.placeholder(dtype=tf.bool, name='phase')
            self.trace_length = tf.placeholder(dtype=tf.int32, name='trace_length')

            batch_size = tf.reshape(tf.shape(self.inputs)[0] // self.trace_length, shape=[])

            self.norm_layer = tf.contrib.layers.batch_norm(self.inputs, scale=True, renorm=True, renorm_decay=renorm_decay, is_training=self.phase)
            self.norm_flat = tf.reshape(self.norm_layer, shape=[batch_size, self.trace_length, n_inputs])

            rnn_cell = tf.contrib.rnn.LSTMCell(num_units=n_inputs, state_is_tuple=True, activation=tf.nn.softsign, use_peepholes=True)
            # TODO: try adding attention to the LSTM
            rnn_cell = tf.contrib.rnn.AttentionCellWrapper(rnn_cell, attention_depth)
            # rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=dropout_keep_prob)
            rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell] * rnn_layers, state_is_tuple=True)

            self.rnn_in = rnn_cell.zero_state(batch_size, dtype=tf.float32)
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(rnn_cell, self.norm_flat, dtype=tf.float32, initial_state=self.rnn_in)
            self.rnn = tf.reshape(self.rnn, shape=tf.shape(self.norm_layer))

            n_hiddens = hidden_units or (n_inputs + n_outputs) // 2
            self.hidden_layer = tf.contrib.layers.fully_connected(self.rnn, n_hiddens, activation_fn=tf.nn.tanh)
            self.output_layer = tf.contrib.layers.fully_connected(self.hidden_layer, n_outputs, activation_fn=None)
            self.probabilities = tf.nn.softmax(self.output_layer)

            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output_layer, labels=self.labels)
            self.loss = tf.reduce_mean(self.losses)
            self.optimizer = tf.train.AdamOptimizer(learn_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

            self.summaries = tf.summary.merge([
                tf.summary.histogram('normed_inputs', self.norm_layer),
                tf.summary.scalar('loss', self.loss),
                tf.summary.histogram('loss_hist', self.losses),
                tf.summary.histogram('output_hist', self.output_layer),
                tf.summary.scalar('max_confidence', tf.reduce_max(self.probabilities))
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
        return sess.run([self.output_layer, self.probabilities, self.rnn_state], feed_dict)

    def update(self, sess, state, labels, trace_length, rnn_state):
        feed_dict = {
            self.inputs: state,
            self.labels: labels,
            self.phase: True,
            self.trace_length: trace_length,
            self.rnn_in: rnn_state
        }

        summaries, global_step, _, loss = sess.run([self.summaries, tf.contrib.framework.get_global_step(),
                                                    self.train_op, self.loss], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)

        return loss

    def compute_loss(self, sess, state, label, rnn_state):
        feed_dict = {
            self.inputs: np.array([state]),
            self.labels: np.array([label]),
            self.phase: False,
            self.trace_length: 1,
            self.rnn_in: rnn_state
        }
        return sess.run(self.loss, feed_dict)
