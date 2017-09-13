import os
import time

import tensorflow as tf


class QEstimator(object):

    def __init__(self,
                 scope: str,
                 n_inputs: int,
                 n_hiddens: int,
                 n_outputs: int,
                 learn_rate: float = 0.0005,
                 decay: float = 0.99,
                 summaries_dir: str = None):
        self.scope = scope

        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, n_inputs], dtype=tf.float32, name='inputs')
            self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name='targets')
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')

            batch_size = tf.shape(self.inputs)[0]

            hidden_layer = tf.contrib.layers.fully_connected(self.inputs, n_hiddens, activation_fn=tf.nn.crelu)
            self.output_layer = tf.contrib.layers.fully_connected(hidden_layer, n_outputs // 2, activation_fn=tf.nn.crelu)
            # Run softmax on output layer to get confidences

            gather_indices = tf.range(batch_size) * tf.shape(self.output_layer)[1] + self.actions
            self.predictions = tf.gather(tf.reshape(self.output_layer, [-1]), gather_indices)

            self.losses = tf.squared_difference(self.targets, self.predictions)
            self.loss = tf.reduce_mean(self.losses)
            self.optimizer = tf.train.RMSPropOptimizer(learn_rate, decay)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

            self.summaries = tf.summary.merge([
                tf.summary.scalar("loss", self.loss),
                tf.summary.histogram("loss_hist", self.losses),
                tf.summary.histogram("q_values_hist", self.output_layer),
                tf.summary.scalar("max_q_value", tf.reduce_max(self.output_layer))
            ])

            self.summary_writer = None
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, 'summaries_{}_{}'.format(scope, time.strftime('%Y%m%d_%H%M%S')))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def predict(self, sess, state):
        return sess.run(self.output_layer, {self.inputs: state})

    def update(self, sess, state, action, target):
        feed_dict = {self.inputs: state, self.targets: target, self.actions: action}
        summaries, global_step, _, loss = sess.run([self.summaries, tf.contrib.framework.get_global_step(),
                                                    self.train_op, self.loss], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)

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
