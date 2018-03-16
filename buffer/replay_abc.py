from collections import deque
import random

class ReplayABC(object):

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

    def record(self, state, action, reward, terminal=False):
        pass


    def size(self):
        return self.count

    def sample_batch(self):
        pass

    def clear(self):
        self.buffer.clear()
        self.count = 0
        self.s, self.a, self.r, self.s2 = None, None, None, None