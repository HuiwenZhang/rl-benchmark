"""
Data structure for implementing experience replay, episode level
"""
import random
from collections import deque

import numpy as np

import replay_abc


class ReplayBufferEpisode(replay_abc.ReplayABC):

    def __init__(self, buffer_size, batch_size, random_seed=1234):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.count = 0
        self.buffer = deque()
        self.s, self.a, self.r, self.s2 = None, None, None, None
        self.episode = []

        random.seed(random_seed)

    def record(self, state, action, reward, next_state, done=False):

        if done == True:
            # add episodic experience to buffer
            # **************an element of a buffer in an episode**************
            if self.count < self.buffer_size:
                self.buffer.append(self.episode)
                self.count += 1
            else:
                self.buffer.popleft()
                self.buffer.append(self.episode)

            # empty the current episode to prepare for the next episode
            self.episode = []

        else:  # not terminal state

            self.episode.append(
                [state, action, reward, next_state, done])

    def size(self):
        return self.count

    def sample_batch(self, sampling_seed=None):
        batch = []

        if self.count < self.batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, self.batch_size)

        #################### for PG ######################
         # batch = [self.buffer[-1]]
        ################################################

        batch = np.array(batch)

        s_batch = []
        a_batch = []
        r_batch = []
        s2_batch = []
        t_batch = []
        for epi in batch:
            tmp_s, tmp_a, tmp_r, tmp_s2, tmp_t= [], [], [], [], []
            for exp in epi:
                tmp_s.append(exp[0])
                tmp_a.append(exp[2])
                tmp_r.append(exp[3])
                tmp_s2.append(exp[5])
                tmp_t.append(exp[7])

            s_batch.append(tmp_s)
            a_batch.append(tmp_a)
            r_batch.append(tmp_r)
            s2_batch.append(tmp_s2)
            t_batch.append(tmp_t)

        return s_batch, a_batch, r_batch, s2_batch, t_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
        self.s, self.a, self.r = None, None, None
