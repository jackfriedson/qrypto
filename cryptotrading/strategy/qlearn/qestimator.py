import os
import time

import tensorflow as tf


class QEstimator(object):

    def __init__(self,
                 scope: str,
                 n_inputs: int,
                 n_outputs: int,
                 hidden_units: int = None,
                 learn_rate: float = 0.0005,
                 optimizer_decay: float = 0.9,
                 renorm_decay: float = 0.9,
                 summaries_dir: str = None):
        self.scope = scope
        n_hiddens = hidden_units if hidden_units is not None else (n_inputs + n_outputs) // 2

        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, n_inputs], dtype=tf.float32, name='inputs')
            self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name='targets')
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
            self.phase = tf.placeholder(dtype=tf.bool, name='phase')

            batch_size = tf.shape(self.inputs)[0]

            norm_layer = tf.contrib.layers.batch_norm(self.inputs, renorm=True, renorm_decay=renorm_decay, is_training=self.phase)
            hidden_layer = tf.contrib.layers.fully_connected(norm_layer, n_hiddens, activation_fn=tf.nn.crelu)
            self.output_layer = tf.contrib.layers.fully_connected(hidden_layer, n_outputs // 2, activation_fn=tf.nn.crelu)
            self.softmax = tf.nn.softmax(self.output_layer)

            gather_indices = tf.range(batch_size) * tf.shape(self.output_layer)[1] + self.actions
            self.predictions = tf.gather(tf.reshape(self.output_layer, [-1]), gather_indices)

            self.losses = tf.squared_difference(self.targets, self.predictions)
            self.loss = tf.reduce_mean(self.losses)
            self.optimizer = tf.train.RMSPropOptimizer(learn_rate, optimizer_decay)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

            self.summaries = tf.summary.merge([
                tf.summary.scalar("loss", self.loss),
                tf.summary.histogram("loss_hist", self.losses),
                tf.summary.histogram("q_values_hist", self.output_layer),
                tf.summary.scalar("max_q_value", tf.reduce_max(self.output_layer)),
                tf.summary.scalar("max_confidence", tf.reduce_max(self.softmax))
            ])

            self.summary_writer = None
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, '{}_{}'.format(scope, time.strftime('%Y%m%d_%H%M%S')))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def predict(self, sess, state, phase):
        return sess.run([self.output_layer, self.softmax], {self.inputs: state, self.phase: phase})

    def update(self, sess, state, action, target):
        feed_dict = {self.inputs: state, self.targets: target, self.actions: action, self.phase: True}
        summaries, global_step, _, loss = sess.run([self.summaries, tf.contrib.framework.get_global_step(),
                                                    self.train_op, self.loss], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)

        return loss

    def compute_loss(self, sess, state, action, target):
        feed_dict = {self.inputs: state, self.targets: target, self.actions: action, self.phase: False}
        loss = sess.run(self.loss, feed_dict)
        return loss


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
