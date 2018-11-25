"""
References: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py
"""

import numpy as np
import random
import copy

class OUNoise:
    def __init__(self, action_size, num_agents, seed, mu=0., theta=0.15, sigma=0.2, epsilon=0.3):
        self.action_size = action_size
        self.mu = mu * np.ones((num_agents, action_size))        
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.epsilon = epsilon
        self.num_agents = num_agents
        
        self.reset(epsilon)

    def reset(self, epsilon):
        self.state = copy.copy(self.mu)
        self.epsilon = epsilon

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.epsilon * self.sigma * np.random.randn(self.num_agents, self.action_size)
        self.state = x + dx
        return self.state