import tensorflow as tf


class QEstimator(object):

    def __init__(self,
                 scope: str,
                 n_inputs: int,
                 n_hiddens: int,
                 n_outputs: int,
                 learn_rate: float = 0.2):
        self.scope = scope

        with tf.variable_scope(scope):
            self.inputs = tf.placeholder(shape=[None, 1, n_inputs], dtype=tf.float32, name='inputs')
            self.targets = tf.placeholder(shape=[None, n_outputs], dtype=tf.float32, name='targets')
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')

            batch_size = tf.shape(self.inputs)[0]

            hidden_layer = tf.contrib.layers.fully_connected(self.inputs, n_hiddens, activation_fn=None)
            self.output_layer = tf.contrib.layers.fully_connected(hidden_layer, n_outputs, activation_fn=None)

            gather_indices = tf.range(batch_size) * tf.shape(self.output_layer)[1] + self.actions
            self.predictions = tf.gather(tf.reshape(self.output_layer, [-1]), gather_indices)

            self.loss = tf.reduce_mean(tf.squared_difference(self.targets, self.predictions))
            self.optimizer = tf.train.AdamOptimizer(learn_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, sess, state):
        return sess.run(self.output_layer, {self.inputs: state})

    def update(self, sess, state, action, target):
        feed_dict = {self.inputs: state, self.targets: target, self.actions: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ModelParametersCopier(object):
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
