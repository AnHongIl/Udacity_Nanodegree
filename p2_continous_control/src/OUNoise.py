import numpy as np
import copy
import random

class OUNoise:
    def __init__(self, action_size, seed, mu=0., theta=0.15, sigma=0.2, epsilon=0.3):
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.epsilon = epsilon
        self.action_size = action_size
        
        self.reset(epsilon)

    def reset(self, epsilon):
        self.state = copy.copy(self.mu)
        self.epsilon = epsilon

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.epsilon * self.sigma * np.random.randn(self.action_size)
        self.state = x + dx
        return self.state