"""
reference: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py
"""

from .model import Actor, Critic
from .OUNoise import OUNoise
from .Memory import ReplayMemory

import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random

class DDPGAgent():
    def __init__(self, state_size, action_size, random_seed, hidden_units, lr_actor, lr_critic, batch_size, gamma, memory_size, epsilon, tau):          
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau        
        
        self.device = torch.device("cuda:0")
        
        self.actor_local = Actor(state_size, action_size, random_seed, hidden_units).to(self.device)
        self.actor_target = Actor(state_size, action_size, random_seed, hidden_units).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        self.critic_local = Critic(state_size, action_size, random_seed, hidden_units).to(self.device)
        self.critic_target = Critic(state_size, action_size, random_seed, hidden_units).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic)
        
        self.noise = OUNoise(action_size, self.seed)
        self.memory = ReplayMemory(self.action_size, self.memory_size, self.batch_size, random_seed)
                
    def reset(self):
        self.noise.reset()
                
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)        

        if len(self.memory.buffer) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
        
    def act(self, states, isNoise):   
        states = torch.from_numpy(states).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()

        if isNoise:
            actions += self.noise.sample()        
        return np.clip(actions, -1., 1.)
        
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)