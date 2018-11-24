"""
Reference: https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8
"""

import numpy as np 
from collections import deque
        
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size): 
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size, 
                               replace=False)
        return [self.buffer[i] for i in idx]    
    
    def unpack(self, experiences):
        states = np.array([each[0] for each in experiences])
        actions = np.array([each[1] for each in experiences])
        rewards = np.array([each[2] for each in experiences])
        next_states = np.array([each[3] for each in experiences])
        dones = np.array([each[4] for each in experiences])
        
        return states, actions, rewards, next_states, dones