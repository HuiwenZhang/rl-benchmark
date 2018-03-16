
"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.
The Cartpole example.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import os
import shutil

# OUTPUT_GRAPH = True
# LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
GLOBAL_NET_SCOPE = 'Global_Net'
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0


class ACNet(object):
    def __init__(
            self,
            sess,
            scope,
            n_in,
            n_out,
            update_global=10,
            gamma=0.9,
            batch_size=32,
            entropy_beta=0.001,
            a_h1_units=100,
            c_h1_units=100,
            a_lr=0.001,
            c_lr=0.001,
            globalAC=None):
        self.sess = sess
        self.n_feature = n_in
        self.n_action = n_out,
        self.update_global = update_global
        self.gamma = gamma
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.a1_units = a_h1_units
        self.c1_units = c_h1_units
        self.batch_size = batch_size
        self.entropy_beta = entropy_beta

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(
                    tf.float32, [
                        None, self.n_feature], 'belief')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(
                    tf.float32, [
                        None, self.n_feature], 'belief')
                self.a = tf.placeholder(tf.int32, [None, 1], 'act')
                self.v_target = tf.placeholder(
                    tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(
                    scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    # log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, self.n_action, dtype=tf.float32), axis=1, keep_dims=True)
                    a_indices = tf.stack(
                        [tf.range(self.batch_size, dtype=tf.int32), self.a], axis=1)
                    log_prob = tf.log(tf.gather_nd(self.a_prob, a_indices))
                    exp_v = log_prob * td
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = self.entropy_beta * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [
                        l_p.assign(g_p) for l_p, g_p in zip(
                            self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [
                        l_p.assign(g_p) for l_p, g_p in zip(
                            self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(
                        zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(
                        zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(
                self.s,
                self.a1_units,
                tf.nn.relu6,
                kernel_initializer=w_init,
                name='la')
            a_prob = tf.layers.dense(
                l_a,
                self.n_action,
                tf.nn.softmax,
                kernel_initializer=w_init,
                name='act_prob')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(
                self.s,
                self.c1_units,
                tf.nn.relu6,
                kernel_initializer=w_init,
                name='lc')
            v = tf.layers.dense(
                l_c,
                1,
                kernel_initializer=w_init,
                name='V(s)')  # state value
        a_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope + '/actor')
        c_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        # local grads applies to global net
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):  # run by a local
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, admissable):  # run by a local
        s = np.array(s)
        s = s[np.newaxis, :]

        # get probabilities for all actions
        prob_on_acts = self.sess.run(self.a_prob, {self.s: s})
        prob_on_acts = prob_on_acts.ravel()
        exec_prob = prob_on_acts[admissable]
        exec_prob /= np.sum(exec_prob)
        action = np.random.choice(admissable, p=exec_prob)
        return np.squeeze(action)


class Worker(object):
    def __init__(
            self,
            name,
            sess,
            scope,
            n_in,
            n_out,
            update_global=10,
            gamma=0.9,
            batch_size=32,
            entropy_beta=0.001,
            a_h1_units=100,
            c_h1_units=100,
            a_lr=0.001,
            c_lr=0.001,
            globalAC=None):
        self.name = name
        self.AC = ACNet(
            name,
            sess,
            scope,
            n_in,
            n_out,
            update_global,
            gamma,
            batch_size,
            entropy_beta,
            a_h1_units,
            c_h1_units,
            a_lr,
            c_lr,
            globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:
                if self.name == 'W_0':
                    self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                if done:
                    r = -5
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % self.update_global == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = self.sess.run(
                            self.AC.v, {
                                self.AC.s: s_[
                                    np.newaxis, :]})[
                            0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + self.gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(
                        buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(
                            0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                    )
                    GLOBAL_EP += 1
                    break


if __name__ == "__main__":
    self.sess = tf.self.session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(self.a_lr, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(self.c_lr, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'Worker_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    self.sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, self.sess.graph)

    worker_threads = []
    for worker in workers:
        def job(): return worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
plt.show()
