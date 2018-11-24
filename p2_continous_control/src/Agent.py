from .Actor import DDPGActor
from .Critic import DDPGCritic
from .OUNoise import OUNoise
from .Memory import Memory

import tensorflow as tf
import numpy as np
import random

class DDPGAgent():
    def __init__(self, state_size, action_size, random_seed, hidden_units, lr_actor, lr_critic, batch_size, gamma, memory_size, num_actors, epsilon, tau, epsilon_decay):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.num_actors = num_actors
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.actor = DDPGActor(state_size, action_size, lr_actor, tau, batch_size, hidden_units)        
        self.critic = DDPGCritic(state_size, action_size, lr_critic, tau, batch_size, hidden_units)        
        
        self.noise = OUNoise(action_size, self.seed, self.epsilon)
        self.memory = Memory(memory_size)
        
    def set_session(self, sess):
        self.sess = sess
        
    def reset(self):
        self.noise.reset(self.epsilon)
        self.critic_loss = []
        
    def step(self, state, action, reward, next_state, done):
        for i in range(self.num_actors):
            self.memory.add((state[i], action[i], reward[i], next_state[i], done[i]))        

        if len(self.memory.buffer) >= self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
        
    def act(self, states, isNoise):   
        actions = self.sess.run(self.actor.actions, feed_dict={self.actor.states: states})
        if isNoise:
            actions += self.noise.sample()        
        return np.clip(actions, -1., 1.)
        
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = self.memory.unpack(experiences)

        next_actions = self.sess.run(self.actor.target_actions, feed_dict={self.actor.target_states: next_states})
        target_Qs = self.sess.run(self.critic.target_Qs, feed_dict={self.critic.target_states: next_states, \
                                                                    self.critic.target_actions: next_actions})

        rewards = np.array(rewards).reshape((self.batch_size, 1))
        dones = np.array(dones).reshape((self.batch_size, 1))
        Ys = rewards + (1 - dones) * self.gamma * target_Qs
            
        loss, _ = self.sess.run([self.critic.loss, self.critic.opt], feed_dict={self.critic.Ys: Ys, \
                                                                           self.critic.states: states, \
                                                                           self.critic.actions: actions})

        predicted_actions = self.sess.run(self.actor.actions, feed_dict={self.actor.states: states})

        critic_grads = self.sess.run(self.critic.grads, feed_dict={self.critic.states: states, \
                                                                   self.critic.actions: predicted_actions})   
        _ = self.sess.run(self.actor.opt, feed_dict={self.actor.states: states, self.actor.critic_grads: critic_grads})
            
        self.sess.run(self.critic.soft_update)                
        self.sess.run(self.actor.soft_update)                

        self.critic_loss.append(loss)
        
        self.soft_update()
        
        self.epsilon -= self.epsilon_decay
    
    def hard_update(self):
        self.sess.run([self.actor.hard_update, self.critic.hard_update])
    
    def soft_update(self):
        self.sess.run([self.actor.soft_update, self.critic.soft_update])             