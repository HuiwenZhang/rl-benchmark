'''
class used for implement actor-critic reinforcement learning
author: alvin
'''

import tensorflow as tf
import numpy as np

"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.
"""

import numpy as np
import tensorflow as tf


class ACNet(object):
    def __init__(
            self,
            sess,
            n_in,
            n_out,
            h1_units=150,
            h2_units=50,
            clr=0.002,
            batch_size=32,
            gamma=0.9,
            alr=0.001):
        self.sess = sess
        self.n_in = n_in
        self.n_out = n_out
        self.batch_size = batch_size
        self.h2_units = h2_units
        self.h1_units = h1_units
        self.gamma = gamma
        self.clr = clr
        self.alr = alr

        self.s = tf.placeholder(tf.float32, [None, self.n_in], name="belief")
        self.s_ = tf.placeholder(tf.float32, [None, self.n_in], "next_belief")
        self.a = tf.placeholder(tf.int32, [None], name="act")
        self.r = tf.placeholder(tf.float32, [None, 1], "reward")
        self.v_ = tf.placeholder(tf.float32, [None, 1], name="next_act")

        w_initializer, b_initializer = tf.random_normal_initializer(
            0., 0.01), tf.constant_initializer(0.)

        share_inputs = self._build_share_net(self.s, 'share_layer')
        with tf.variable_scope('Actor'):
            self.act_probs = tf.layers.dense(
                inputs=share_inputs,
                units=self.n_out,    # output units
                activation=tf.nn.softmax,
                kernel_initializer=w_initializer,  # weights
                bias_initializer=b_initializer,  # biases
                name='policy'
            )

        with tf.variable_scope('Critic'):
            self.v = tf.layers.dense(
                inputs=share_inputs,
                units=1,    # output units
                kernel_initializer=w_initializer,  # weights
                bias_initializer=b_initializer,  # biases
                name='V_s'
            )

        with tf.variable_scope('closs'):
            # advantage
            self.td_error = self.r + self.gamma * self.v_ - self.v
            # TD_error = (r+gamma*V_next) - V_eval
            self.closs = tf.reduce_mean(tf.square(self.td_error))
        with tf.variable_scope('ctrain'):
            self.ctrain_op = tf.train.AdamOptimizer(
                self.clr).minimize(self.closs)
            # self.ctrain_op = tf.train.GradientDescentOptimizer(self.clr).minimize(self.closs)
            # self.ctrain_op = tf.train.AdadeltaOptimizer(self.clr).minimize(self.closs)
            # self.ctrain_op = tf.train.RMSPropOptimizer(self.clr).minimize(self.closs)

        with tf.variable_scope('aloss'):
            idx = tf.stack(
                [tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            log_prob = tf.log(tf.gather_nd(self.act_probs, idx))
            # if the current act is advantage (Q(s,a) - v(s)), then give it
            # bigger probality, so maximize exp_v
            # advantage (TD_error) guided loss
            self.aloss = -tf.reduce_mean(log_prob * self.td_error)
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(self.alr).minimize(
                self.aloss)  # minimize(-exp_v) = maximize(exp_v)
            # self.atrain_op = tf.train.GradientDescentOptimizer(self.alr).minimize(-self.exp_v)
            # self.atrain_op = tf.train.AdadeltaOptimizer(self.alr).minimize(-self.exp_v)
            # self.atrain_op = tf.train.RMSPropOptimizer(self.alr).minimize(-self.exp_v)

    def _build_share_net(self, inputs, scope):
        with tf.variable_scope(scope):
            w_initializer, b_initializer = tf.random_normal_initializer(
                0., 0.01), tf.constant_initializer(0.)
            l1 = tf.layers.dense(
                inputs,
                units=self.h1_units,
                activation=tf.nn.relu,
                kernel_initializer=w_initializer,
                bias_initializer=b_initializer,
                name='share_layer1')
            return tf.layers.dense(
                l1,
                units=self.h2_units,
                activation=tf.nn.relu,
                kernel_initializer=w_initializer,
                bias_initializer=b_initializer,
                name='share_layer2')

    def train(self, s, a, r, s_):
        # train in batch, sample act in one state
        # s = s[np.newaxis, :]
        v_ = self.sess.run(self.v, feed_dict={self.s: s_})
        self.sess.run([self.closs, self.aloss, self.atrain_op, self.ctrain_op],
                      feed_dict={self.s: s, self.a: a, self.r: r, self.v_: v_})
        return

    def choose_action(self, s, admissable):
        s = np.array(s)
        s = s[np.newaxis, :]

        # get probabilities for all actions
        prob_on_acts = self.sess.run(self.act_probs, {self.s: s})
        prob_on_acts = prob_on_acts.ravel()
        exec_prob = prob_on_acts[admissable]
        exec_prob /= np.sum(exec_prob)
        action = np.random.choice(admissable, p=exec_prob)
        return np.squeeze(action)

    def predict_action(self, s):
        s = np.array(s)
        s = s[np.newaxis, :]
        act_probs = self.sess.run(self.act_probs, feed_dict={self.s: s})
        return self.sess.run(tf.log(act_probs, name='retriecve_log_prob'))

    def predict_value(self, s):
        s = np.array(s)
        s = s[np.newaxis, :]
        return self.sess.run(self.v, feed_dict={self.s: s})

    def load_network(self, load_filename):
        self.saver = tf.train.Saver()
        try:
            self.saver.restore(self.sess, load_filename)
            print "Successfully loaded:", load_filename
        except BaseException:
            print "Could not find old network weights"

    def save_network(self, save_filename):
        print 'Saving a2c-network...'
        self.saver.save(self.sess, save_filename)
