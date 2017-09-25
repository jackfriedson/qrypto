import time

import numpy as np
import tensorflow as tf


class QEstimator(object):

    def __init__(self,
                 scope: str,
                 rnn_cell,
                 n_inputs: int,
                 n_outputs: int,
                 hidden_units: int = None,
                 learn_rate: float = 0.0005,
                 optimizer_decay: float = 0.9,
                 renorm_decay: float = 0.9,
                 summaries_dir: str = None):
        self.scope = scope

        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, n_inputs], dtype=tf.float32, name='inputs')
            self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name='targets')
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
            self.phase = tf.placeholder(dtype=tf.bool, name='phase')
            self.trace_length = tf.placeholder(dtype=tf.int32, name='trace_length')

            batch_size = tf.shape(self.inputs)[0]
            rnn_batch_size = tf.reshape(batch_size // self.trace_length, shape=[])

            self.norm_layer = tf.contrib.layers.batch_norm(self.inputs, renorm=True, renorm_decay=renorm_decay, is_training=self.phase)
            self.norm_flat = tf.reshape(self.norm_layer, shape=[rnn_batch_size, self.trace_length, n_inputs])

            self.rnn_in = rnn_cell.zero_state(rnn_batch_size, dtype=tf.float32)
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(rnn_cell, self.norm_flat, dtype=tf.float32, initial_state=self.rnn_in)
            self.rnn = tf.reshape(self.rnn, shape=tf.shape(self.norm_layer))

            # self.hidden_layer = tf.contrib.layers.fully_connected(self.rnn, n_hiddens, activation_fn=tf.nn.crelu)
            self.output_layer = tf.contrib.layers.fully_connected(self.rnn, n_outputs, activation_fn=None, biases_initializer=None)
            self.softmax = tf.nn.softmax(self.output_layer)

            gather_indices = tf.range(batch_size) * tf.shape(self.output_layer)[1] + self.actions
            self.predictions = tf.gather(tf.reshape(self.output_layer, [-1]), gather_indices)

            self.losses = tf.squared_difference(self.targets, self.predictions)

            # Mask first half of losses
            maskA = tf.zeros([rnn_batch_size, self.trace_length // 2])
            maskB = tf.ones([rnn_batch_size, self.trace_length // 2])
            mask = tf.reshape(tf.concat([maskA, maskB], 1), [-1])
            self.masked_loss = tf.reduce_mean(tf.multiply(self.losses, mask))
            self.unmasked_loss = tf.reduce_mean(self.losses)
            self.optimizer = tf.train.AdamOptimizer(learn_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.masked_loss, global_step=tf.contrib.framework.get_global_step())

            self.summaries = tf.summary.merge([
                tf.summary.scalar('loss', self.masked_loss),
                tf.summary.histogram('loss_hist', self.losses),
                tf.summary.histogram('q_values_hist', self.output_layer),
                tf.summary.scalar('max_q_value', tf.reduce_max(self.output_layer)),
                tf.summary.scalar('max_confidence', tf.reduce_max(self.softmax))
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
        return sess.run([self.output_layer, self.softmax, self.rnn_state], feed_dict)

    def update(self, sess, state, action, target, trace_length, rnn_state):
        feed_dict = {
            self.inputs: state,
            self.targets: target,
            self.actions: action,
            self.phase: True,
            self.trace_length: trace_length,
            self.rnn_in: rnn_state
        }

        summaries, global_step, _, loss = sess.run([self.summaries, tf.contrib.framework.get_global_step(),
                                                    self.train_op, self.masked_loss], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)

        return loss

    def compute_loss(self, sess, state, action, target, rnn_state):
        feed_dict = {
            self.inputs: np.array([state]),
            self.targets: np.array([target]),
            self.actions: np.array([action]),
            self.phase: False,
            self.trace_length: 1,
            self.rnn_in: rnn_state
        }
        return sess.run(self.unmasked_loss, feed_dict)

class ModelParametersCopier():
    def __init__(self, estimator_from, estimator_to):
        from_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator_from.scope)]
        from_params = sorted(from_params, key=lambda v: v.name)
        to_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator_to.scope)]
        to_params = sorted(to_params, key=lambda v: v.name)

        self.update_ops = []
        for from_v, to_v in zip(from_params, to_params):
            op = to_v.assign(from_v)
            self.update_ops.append(op)

    def make(self, sess):
        sess.run(self.update_ops)
