import tensorflow as tf
import numpy as np
from collections import deque

from .DQN import Network
from .Memory import Memory

class Agent():
    def __init__(self, state_size, action_size, hidden_size, memory_size, batch_size, explore_stop, decay_rate, gamma, learning_rate, tau, update_every):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size        
        
        self.explore_start = 1.0             
        self.explore_stop = explore_stop            
        self.decay_rate = decay_rate    
        self.update_every = update_every
        
        self.gamma = gamma
                
        self.dqn = Network(state_size, action_size, hidden_size, learning_rate, 'local')
        self.target_dqn = Network(state_size, action_size, hidden_size, learning_rate, 'target')
                
        self.l_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'local')            
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'target')    
                
        self.hard_update_op = [tf.assign(t, l) for t, l in zip(self.t_params, self.l_params)]        
        self.soft_update_op = [tf.assign(t, (1-tau) * t + tau * l) for t, l in zip(self.t_params, self.l_params)]        
        
        self.memory = Memory(memory_size)
        
        self.t = 0
          
    def set_session(self, sess):
        self.sess = sess
        
    def reset(self):
        self.loss = []
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))
        
        if len(self.memory.buffer) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)        
                
        self.t += 1
        if self.t % self.update_every == 0:
            self.hard_update()
        else:
            self.soft_update()
                
    def act(self, state, isTraining):
        if isTraining is False:
            if 0.05 > np.random.rand():
                action = np.random.choice(self.action_size)
            else:
                action = np.asscalar(np.argmax(self.sess.run(self.dqn.Qs, feed_dict={self.dqn.states: state.reshape(1, self.state_size)})))            
            return action
        
        explore_p = self.explore_stop + (self.explore_start - self.explore_stop)*np.exp(-self.decay_rate * self.t) 
        if explore_p > np.random.rand():
            action = np.random.choice(self.action_size)
        else:
            Qs = self.sess.run(self.dqn.Qs, feed_dict={self.dqn.states: state.reshape(1, self.state_size)})
            action = np.asscalar(np.argmax(Qs))
        return action, explore_p
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = self.memory.unpack(experiences)

        target_next_Qs = self.sess.run(self.target_dqn.Qs, feed_dict={self.target_dqn.states: next_states})
        target_next_actions = np.argmax(target_next_Qs, 1)
        
        target_Q = self.sess.run(self.target_dqn.Q, feed_dict={self.target_dqn.states: next_states, self.target_dqn.actions: target_next_actions})
        
        rewards = rewards.reshape((self.batch_size, 1))
        dones = dones.reshape((self.batch_size, 1))
        Ys = rewards + (1 - dones) * self.gamma * target_Q

        loss, _ = self.sess.run([self.dqn.loss, self.dqn.opt], feed_dict={self.dqn.Ys: Ys, \
                                                                          self.dqn.states: states, \
                                                                          self.dqn.actions: actions})
        self.loss.append(loss)

    def hard_update(self):
        self.sess.run(self.hard_update_op)

    def soft_update(self):
        self.sess.run(self.soft_update_op)