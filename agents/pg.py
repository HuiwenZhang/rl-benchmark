import tensorflow as tf
import numpy as np
import os
import cPickle as pickle
import copy

# author:alvin


class PolicyGradient(object):
    def __init__(self,
                 sess, n_in, n_action, clip_grad=1,
                 baseline=True,
                 learning_rate=0.01,
                 gamma=0.95,
                 h1_units=120,
                 h2_units=60,
                 out_graph=False,
                 bias_lr=0.001,
                 buffer=None
                 ):
        self.sess = sess
        self.a_dim = n_action
        self.s_dim = n_in
        self.lr = learning_rate
        self.gamma = gamma
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.clip_grad = clip_grad
        self.baseline = baseline
        self.blr = bias_lr
        self.out_graph = out_graph
        self.buffer = buffer
        self.alg = 'pg'

        # placeholders
        self.s = tf.placeholder(
            tf.float32, [
                None, self.s_dim], name='observation')
        self.a = tf.placeholder(tf.int32, [None], name='action')
        self.disr = tf.placeholder(
            tf.float32, [None], name='discount_episode_rewards')

        # build nets, both eval_net and target net
        self.bias = tf.squeeze(
            self._approximate_base_func('baseline_func'), axis=1)
        bias_loss = tf.squared_difference(
            self.disr, self.bias, name='bias_err')
        self.bias_train = tf.train.AdamOptimizer(self.blr).minimize(bias_loss)

        self.act_probs = self._build_net('policy_net')  # build policy net
        params = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope='policy_net')

        with tf.variable_scope('loss'):
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
            neg_log_prob = tf.reduce_sum(-tf.log(self.act_probs)
                                         * tf.one_hot(self.a, self.a_dim), axis=1)
            if self.baseline:
                self.loss = tf.reduce_mean(
                    neg_log_prob * (self.disr - self.bias))
            else:
                self.loss = tf.reduce_mean(neg_log_prob * self.disr)
        if self.clip_grad > 0:
            opt = tf.train.AdamOptimizer(self.lr)
            # opt = tf.train.RMSPropOptimizer(self.lr)
            gvs = opt.compute_gradients(self.loss, params)
            grad = [(tf.clip_by_value(grad, -self.clip_grad, self.clip_grad), var)
                    for grad, var in gvs]
            self.train_op = opt.apply_gradients(grad)
        else:
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        if self.out_graph:
            if not os.path.exists('tblogs/pglog'):
                os.makedirs('tblogs/pglog')
            tf.summary.FileWriter('tblogs/pglog', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _approximate_base_func(self, scope):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(self.s, self.h1_units, tf.nn.relu)
            l2 = tf.layers.dense(l1, self.h2_units, tf.nn.relu)
            baseline = tf.layers.dense(l2, 1, name='estimated_baseline')
        return baseline

    def _build_net(self, scope):
        init_w, init_b = tf.random_normal_initializer(
            0., 0.01), tf.constant_initializer(0.)
        with tf.variable_scope(scope):
            e1 = tf.layers.dense(
                self.s,
                self.h1_units,
                tf.nn.relu,
                kernel_initializer=init_w,
                bias_initializer=init_b,
                name='layer1')
            e2 = tf.layers.dense(
                e1,
                self.h2_units,
                tf.nn.relu,
                kernel_initializer=init_w,
                bias_initializer=init_b,
                name='layer2')
            return tf.layers.dense(
                e2,
                self.a_dim,
                tf.nn.softmax,
                kernel_initializer=init_w,
                bias_initializer=init_b,
                name='probs')

    def _build_policy_net(self, scope):
        with tf.variable_scope(scope):
            W_fc1 = tf.Variable(tf.truncated_normal(
                [self.s_dim, self.h1_units], stddev=0.01))
            b_fc1 = tf.Variable(tf.zeros([self.h1_units]))
            h_fc1 = tf.nn.relu(tf.matmul(self.s, W_fc1) + b_fc1)

            W_fc2 = tf.Variable(tf.truncated_normal(
                [self.h1_units, self.h2_units], stddev=0.01))
            b_fc2 = tf.Variable(tf.zeros([self.h2_units]))
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

            W_out = tf.Variable(tf.truncated_normal(
                [self.h2_units, self.a_dim], stddev=0.01))
            b_out = tf.Variable(tf.zeros([self.a_dim]))
            prob = tf.nn.softmax(tf.matmul(h_fc2, W_out) + b_out)

        return prob

    def discount_rewards(self, ep_rewards):
        """
        retrieve episode rewards
        :param ep_rewards:
        :return:
        """
        ep_rewards = np.array(ep_rewards).squeeze()
        discount_ep_rs = np.zeros_like(ep_rewards)
        temp = 0
        for step in reversed(range(len(ep_rewards))):
            temp = temp * self.gamma + ep_rewards[step]
            discount_ep_rs[step] = temp

        # normalize
        # discount_ep_rs -= np.mean(discount_ep_rs)
        # discount_ep_rs /= np.std(discount_ep_rs)
        return discount_ep_rs.tolist()

    def train(self, inputs, actions, discount_r):

        # discount_norm_rewards = self._discount_norm_rewards(rewards)

        self.sess.run(
            [self.train_op, self.bias_train],
            feed_dict={
                self.s: inputs,
                self.a: actions,
                self.disr: discount_r,
            })

        return

    def choose_action(self, beliefVec):
        inputs = np.array(beliefVec)
        if inputs.ndim < 2:
            inputs = inputs[np.newaxis, :]

        probs = self.sess.run(
            self.act_probs, feed_dict={
                self.s: inputs})  # probs
        return np.random.choice(range(self.a_dim), p=np.squeeze(probs))


    def load_network(self, load_filename):
        try:
            self.saver.restore(self.sess, load_filename)
            print "Successfully load weigths form:", load_filename
        except BaseException:
            print "Could not find old network weigths"

    def savePolicy(self, policyDir):
        """
        :param policyDir: folder used to save policy
        :return:
        """
        print 'Saving dqn network....'
        pol_file_name = policyDir + '.dqn.ckpt'
        dir_name = os.path.dirname(pol_file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        self.saver.save(self.sess, pol_file_name)

        # save buffer
        f = open(policyDir + '.dqn.buffer', 'wb')
        pickle.dump(self.buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def loadPolicy(self, saveDir):
        """
        load models and buffer
        """
        # load models
        self.load_network(saveDir + 'dqn.ckpt')

        # load replay buffer
        try:
            print 'load from: ', saveDir
            f = open(saveDir + 'dqn.buffer', 'rb')
            buffer = pickle.load(f)
            self.buffer = copy.deepcopy(buffer)
            print 'Loading both model and buffer from %s ...' % saveDir
            f.close()
        except BaseException:
            print "Loading only models..."
