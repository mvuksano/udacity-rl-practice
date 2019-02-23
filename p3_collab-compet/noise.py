# https://www.quora.com/Why-do-we-use-the-Ornstein-Uhlenbeck-Process-in-the-exploration-of-DDPG

import random
import copy
import numpy as np

class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    """
    def __init__(self, size, mu=0., theta=0.15, sigma=0.1):
        """
        Initialise parameters and noise process.
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def sample(self):
        """
        Update internal state and return as noise sample.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
        
    def reset(self):
        """
        Reset internal state (=noise)  to mean (=mu).
        """
        self.state = copy.copy(self.mu)