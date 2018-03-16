import tensorflow as tf
import tensorflow.contrib.layers as layers


def _make_network(hiddens, inpt, num_actions, scope='network', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for i, hidden in enumerate(hiddens):
            out = tf.layers.dense(out, hidden, name='d{}'.format(i),
                bias_initializer=tf.constant_initializer(0.),
                kernel_initializer=tf.random_normal_initializer(0.0, 0.01))
            out = tf.nn.relu(out)

        # policy branch
        mu = tf.layers.dense(
            out,
            num_actions,
            kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
            name='mu',
        )
        policy = tf.nn.softmax(mu + 1e-5)
        # value branch
        value = tf.layers.dense(
            out,
            1,
            kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
            name='Q'
        )
    return policy, value

def make_network(hiddens):
    return lambda *args, **kwargs: _make_network(hiddens, *args, **kwargs)

import tensorflow as tf


def _make_network_options(hiddens, inpt, num_actions, scope='network', reuse=None, n_opt=0):
    option = tf.placeholder(tf.int32, [None], 'option')
    idx = tf.stack([tf.range(tf.shape(option)[0]), option], axis=1)
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for i, hidden in enumerate(hiddens):
            out = tf.layers.dense(out, hidden, name='d{}'.format(i),
                bias_initializer=tf.constant_initializer(0.),
                kernel_initializer=tf.random_normal_initializer(0.0, 0.01))
            out = tf.nn.relu(out)

        opts = []
        values = []
        for i in range(n_opt):
            # policy branch
            mu = tf.layers.dense(
                out,
                num_actions,
                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                name='mu_{}'.format(i),
            )
            policy = tf.nn.softmax(mu + 1e-5)
            opts.append(policy)
            # value branch
            value = tf.layers.dense(
                out,
                1,
                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                name='Q_{}'.format(i)
            )
            values.append(value)
        opts = tf.stack(opts, axis=1)
        policy = tf.gather_nd(opts, idx)
        vls = tf.stack(values, axis=1)
        value = tf.gather_nd(vls, idx)

    return option, policy, value

def make_network_options(hiddens):
    return lambda *args, **kwargs: _make_network_options(hiddens, *args, **kwargs)

