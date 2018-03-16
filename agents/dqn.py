import tensorflow as tf
import numpy as np
import os
import cPickle as pickle
import copy
# author:alvin


class DQN(object):
    def __init__(self, sess, s_dim, a_dim,
                 learning_rate=0.01,
                 gamma=0.95,
                 batch_size=32,
                 h1_units=120,
                 h2_units=30,
                 out_graph=False,
                 double_dqn=False,
                 clip_loss=1,
                 replay_type='vanilla',
                 tau=0.05,
                 buffer=None,
                 inpolicy_file=None,
                 epsilon=0.9
                 ):
        self.sess = sess
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.ddqn = double_dqn
        self.clip = clip_loss
        self.replay_type = replay_type
        self.out_graph = out_graph
        self.buffer = buffer
        self.epsilon = epsilon
        self.inpolicy_file = inpolicy_file
        self.alg = 'dqn'

        self.s = tf.placeholder(tf.float32, [None, self.s_dim], name='state')
        self.s_ = tf.placeholder(tf.float32, [None, self.s_dim], name='state_')
        self.a = tf.placeholder(tf.int32, [None], name='action')
        self.q_target = tf.placeholder(tf.float32, [None, 1], name='target_q')

        if self.replay_type == 'prioritized':
            self.weights = tf.placeholder(
                tf.float32, [None], name='IS_weights')

        # build nets, both eval_net and target net
        self.q = self._build_net(self.s, 'eval_net')
        self.q_ = self._build_net(self.s_, 'target_net')

        # find the index for the action we actually acted on, then update Q
        # value wrt this action
        self.qeval_a = tf.reduce_sum(
            tf.multiply(
                self.q,
                tf.one_hot(
                    self.a,
                    self.a_dim)),
            axis=1,
            keep_dims=True)

        with tf.variable_scope('loss'):
            td_err = self.q_target - self.qeval_a
            self.abs_err = tf.abs(td_err)
            # if self.replay_type == 'prioritized':
            #     td_err = tf.reduce_mean(self.weights * td_err, name='Weighted_TD_ERR')
            # self.loss = tf.reduce_mean(self.clip_err(td_err), name='loss')
            self.loss = tf.reduce_mean(
                self.huber_loss(
                    td_err,
                    self.clip),
                name='clip_loss')

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            # self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
            # self.train_op = tf.train.AdadeltaOptimizer(self.lr).minimize(self.loss)
            # self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        t_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='target_net')
        e_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='eval_net')
        #self.saver = tf.train.Saver()

        with tf.variable_scope('update_target'):
            if tau is None:
                self.update_target_op = [
                    tf.assign(
                        t, e) for t, e in zip(
                        t_params, e_params)]
            else:
                self.update_target_op = [
                    t_params[i].assign(
                        tf.scalar_mul(
                            tau,
                            e_params[i]) +
                        tf.scalar_mul(
                            1 -
                            tau,
                            t_params[i])) for i in range(
                        len(t_params))]

        if self.out_graph:
            if not os.path.exists('tblogs/dqnlog'):
                os.makedirs('tblogs/dqnlog')
            tf.summary.FileWriter('tblogs/dqnlog', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        if self.inpolicy_file is not None:
            self.loadPolicy(self.inpolicy_file)

    def _build_net(self, inputs, scope, reuse=None):
        '''
        build three layer fully connected net
        :param inputs:
        :param scope:
        :param reuse:
        :return:
        '''
        with tf.variable_scope(scope, reuse=reuse):
            W_fc1 = tf.Variable(tf.truncated_normal(
                [self.s_dim, self.h1_units], stddev=0.01))
            b_fc1 = tf.Variable(tf.zeros([self.h1_units]))
            h_fc1 = tf.nn.relu(tf.matmul(inputs, W_fc1) + b_fc1)

            W_fc2 = tf.Variable(tf.truncated_normal(
                [self.h1_units, self.h2_units], stddev=0.01))
            b_fc2 = tf.Variable(tf.zeros([self.h2_units]))
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

            W_out = tf.Variable(tf.truncated_normal(
                [self.h2_units, self.a_dim], stddev=0.01))
            b_out = tf.Variable(tf.zeros([self.a_dim]))
            Qout = tf.matmul(h_fc2, W_out) + b_out
            return Qout

    def train(self, inputs, actions, target_q, weights=None):
        # qe = self.sess.run(self.q, feed_dict={self.s:inputs, self.a: actions,self.r:rewards, self.s_: next_states})

        if self.replay_type == 'prioritized':
            _, cost, abs_err = self.sess.run(
                [self.train_op, self.loss, self.abs_err],
                feed_dict={
                    self.s: inputs,
                    self.a: actions,
                    self.q_target: target_q,
                    self.weights: weights
                })
        else:
            _, cost, abs_err = self.sess.run(
                [self.train_op, self.loss, self.abs_err],
                feed_dict={
                    self.s: inputs,
                    self.a: actions,
                    self.q_target: target_q})
        return cost, abs_err

    def predict_value(self, inputs):
        # every time for one single state, output single action
        inputs = np.array(inputs)
        if inputs.ndim < 2:
            inputs = inputs[np.newaxis, :]
        q_eval, q_ = self.sess.run([self.q, self.q_], feed_dict={
                                   self.s: inputs, self.s_: inputs})
        return q_eval, q_

    def update_target(self):
        print '\ntarget_params_replaced\n'
        self.sess.run(self.update_target_op)

    def choose_action(self, input):
        # decide the current action based on the current max Q
        if input.ndim < 2:
            input = input[np.newaxis, :]
        q_value = self.sess.run(self.q, feed_dict={self.s: input})
        action = np.argmax(q_value)
        if np.random.uniform() < self.epsilon:
            return action
        else:
            return np.random.randint(0, self.a_dim)

    def compute_target_q(self, s, a, r, s_, t):
        '''
        compute target Q value in batch mode
        :param s:
        :param a:
        :param r:
        :param s_:
        :param s_ori:
        :param t:
        :return:
        '''
        q_eval_current, q_tar_current = self.predict_value(s)
        q_eval_next, q_tar_next = self.predict_value(s_)
        y = []
        for i in range(self.batch_size):
            Qi = 0
            if t[i]:
                Qi = r[i]
            else:
                action_Q = q_tar_next[i]
                if not self.ddqn:
                    Qi = r[i] + self.gamma * np.max(action_Q)
                else:
                    eval_Q = q_eval_next[i]
                    Qi = r[i] + self.gamma * action_Q[np.argmax(eval_Q)]
            y.append(Qi)

            # update weigths
            # td_err = np.abs(q_eval_current[i, a[i]] - Qi)
            # self.episodes[self.domainString].update(idx[i], td_err)
        return np.array(y)

    def clip_err(self, x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

    def huber_loss(self, err, d):
        return tf.where(tf.abs(err) < d, 0.5 * tf.square(err),
                        d * (tf.abs(err) - d / 2))

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
