"""
Data structure for sample based algorithms

"""
import random
from collections import deque

import numpy as np

import buffer.replay_abc as replay_abc


class ReplayBuffer(replay_abc.ReplayABC):

    def __init__(self, buffer_size, batch_size, random_seed=1234):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.count = 0
        self.buffer = deque()
        self.s, self.a, self.r, self.s2 = None, None, None, None

        random.seed(random_seed)

    def record(self, state, action, reward, next_state, done=False):
        # reward = float(reward)
        """
        record experience:
        current state, current action, reward, next state and next action, done

        """

        self.s, self.a, self.r, self.s2, self.done = state, action, reward, next_state, done

        # notice here the experience just like a link, they overide at the last end and the next begin - alvin
        experience = [self.s, self.a, self.r,  self.s2, self.done]

        # add experience to buffer
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)



    def size(self):
        return self.count

    def sample(self):
        batch = []

        """
        s_batch  = np.array([_[0] for _ in self.buffer])
        a_batch  = np.array([_[1] for _ in self.buffer])
        r_batch  = np.array([_[2] for _ in self.buffer])
        s2_batch = np.array([_[3] for _ in self.buffer])
        t_batch  = np.array([_[4] for _ in self.buffer])
        print s_batch
        print a_batch
        print r_batch
        print s2_batch
        print t_batch
        """
        if self.count < self.batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, self.batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s2_batch = np.array([_[3] for _ in batch])
        t_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch,  s2_batch, t_batch




    def clear(self):
        self.buffer.clear()
        self.count = 0
        self.s, self.a, self.r = None, None, None
