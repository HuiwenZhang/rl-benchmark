"""
A simple version of Proximal Policy Optimization (PPO) using single thread.


"""
from policy.DRL.ppo_utils import network
import tensorflow as tf
import numpy as np


METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    # Clipped surrogate objective, find this is better
    dict(name='clip', epsilon=0.1),
][1]        # choose the method for optimization


class PPO(object):

    def __init__(
            self,
            sess,
            n_feature,
            n_acton,
            gamma=0.9,
            lr=1e-4,
            beta=0.01,
            a_update_steps=10,
            c_update_steps=10,
            h1_units=200,
            h2_units=100,
            out_graph=False,
            scope=None,
            n_options=0):
        self.sess = sess
        self.s_dim = n_feature
        self.a_dim = n_acton
        self.gamma = gamma
        self.lr = lr
        self.a_update_step = a_update_steps  # update times for policy and evaluat
        self.c_update_step = c_update_steps
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.out_graph = out_graph
        self.scope = scope
        self.n_options = n_options

        self.tfs = tf.placeholder(tf.float32, [None, self.s_dim], 'belief')
        self.tfa = tf.placeholder(tf.int32, [None], 'action')
        # advantage computed by general advantage estimate
        self.gae = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # target value used to update value net
        self.target_value = tf.placeholder(tf.float32, [None, 1], 'value')

        with tf.variable_scope(self.scope):
            if self.n_options > 0:
                net = network.make_network_options(
                    [self.h1_units, self.h2_units])
                self.option, pi, self.v = net(
                    self.tfs, self.a_dim, scope='network', reuse=None, n_opt=self.n_options)
                self.old_option, oldpi, oldv = net(
                    self.tfs, self.a_dim, scope='network_old', reuse=None, n_opt=self.n_options)
            else:
                net = network.make_network([self.h1_units, self.h2_units])
                pi, self.v = net(
                    self.tfs, self.a_dim, scope='network', reuse=None)
                oldpi, oldv = net(
                    self.tfs, self.a_dim, scope='network_old', reuse=None)
            pi_params = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=self.scope + '/network')
            oldpi_params = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=self.scope + '/network_old')
            self.all_act_prob = pi
            self.tfadv = self.gae - self.v  # advantage

            # surrogate loss
            pi_a = tf.reduce_sum(
                pi *
                tf.one_hot(
                    self.tfa,
                    self.a_dim),
                axis=1,
                keep_dims=True)
            oldpi_a = tf.reduce_sum(
                oldpi *
                tf.one_hot(
                    self.tfa,
                    self.a_dim),
                axis=1,
                keep_dims=True)
            ratio = pi_a / oldpi_a

            surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                surrogate = -tf.reduce_mean(tf.minimum(surr,
                                                       tf.clip_by_value(ratio,
                                                                        1. - METHOD['epsilon'],
                                                                        1. + METHOD['epsilon']) * self.tfadv))

            with tf.variable_scope('loss'):
                # td error loss/ value loss
                value_loss = tf.reduce_mean(
                    tf.square(self.target_value - self.v))

                # entropy loss, encourage exploration
                entropy = -tf.reduce_sum(pi * tf.log(pi))
                penalty = -beta * entropy

                # total loss
                loss = surrogate + value_loss + penalty

                # optimize operations
                optimizer = tf.train.AdamOptimizer(self.lr)
                self.train_op = optimizer.minimize(loss, var_list=pi_params)

            with tf.variable_scope('update_oldpi'):
                self.update_oldpi_op = [
                    oldp.assign(p) for p, oldp in zip(
                        pi_params, oldpi_params)]

            tf.summary.scalar('critic_loss', loss)

    def train(self, s, a, advantage, value):
        advantage = np.array(advantage)[:, np.newaxis]
        value = np.array(value)[:, np.newaxis]
        self.sess.run(self.update_oldpi_op)
        if self.n_options > 0:
            inputs, opt = self._process_inputs_option(s)
            [self.sess.run(self.train_op,
                           {self.tfs: inputs,
                            self.tfa: a,
                            self.gae: advantage,
                            self.target_value: value,
                            self.option: opt,
                            self.old_option: opt}) for _ in range(self.a_update_step)]
        else:
            [self.sess.run(self.train_op,
                           {self.tfs: s,
                            self.tfa: a,
                            self.gae: advantage,
                            self.target_value: value}) for _ in range(self.a_update_step)]

    def choose_action(self, beliefVec):
        inputs = np.array(beliefVec)
        if inputs.ndim < 2:
            inputs = inputs[np.newaxis, :]
        if self.n_options > 0:
            inputs, opts = self._process_inputs_option(beliefVec)
            probs = self.sess.run(
                self.all_act_prob, feed_dict={
                    self.tfs: inputs, self.option: opts})
        else:
            probs = self.sess.run(
                self.all_act_prob, feed_dict={
                    self.tfs: inputs})  # probs
        return np.squeeze(probs)  # remove dim is 1

    def predict(self, s):
        s = np.array(s)
        if s.ndim < 2:
            s = s[np.newaxis, :]
        if self.n_options < 1:
            return self.sess.run(self.v, {self.tfs: s})
        else:
            inputs, opt = self._process_inputs_option(s)
            return self.sess.run(
                self.v, feed_dict={
                    self.tfs: inputs, self.option: opt})

    def load_network(self, load_filename):
        self.saver = tf.train.Saver()
        try:
            self.saver.restore(self.sess, load_filename)
            print "Successfully load weigths form:", load_filename
        except BaseException:
            print "Could not find old network weigths"

    def _discount_norm_rewards(self, ep_rewards):
        ep_rewards = np.array(ep_rewards).squeeze()
        discount_ep_rs = np.zeros_like(ep_rewards)
        temp = 0
        for step in reversed(range(len(ep_rewards))):
            temp = temp * self.gamma + ep_rewards[step]
            discount_ep_rs[step] = temp

        # normalize
        discount_ep_rs -= np.mean(discount_ep_rs)
        discount_ep_rs /= np.std(discount_ep_rs)
        return np.matrix(discount_ep_rs).T

    def save_network(self, save_filename):
        print "Saving dqn-network....."
        self.saver.save(self.sess, save_filename)

    def _process_inputs_option(self, inputs):
        features = inputs[:, :self.s_dim]
        options_onehot = inputs[:, self.s_dim:]
        options = np.argmax(options_onehot, axis=1)
        return features, options
