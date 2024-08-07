import random
import copy
import numpy as np


class OUNoise:

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        random.seed(seed)
        np.random.seed(seed)
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.size)
        self.state = x + dx
        return self.state
