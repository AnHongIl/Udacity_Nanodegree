from .Actor import DDPGActor
from .Critic import DDPGCritic
from .Memory import Memory
from .OUNoise import OUNoise

import random
import numpy as np

class DDPGAgents():
    def __init__(self, state_size, action_size, random_seed, memory_size, hidden_units, lr_actor, lr_critic, batch_size, gamma, num_agents, epsilon, tau, epsilon_decay):
        self.state_size = state_size
        self.action_size = action_size

        self.seed = random.seed(random_seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.num_agents = num_agents
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.memories = [Memory(memory_size) for i in range(num_agents)]
        
        self.actors = [DDPGActor(i, state_size, action_size, lr_actor, tau, batch_size, hidden_units) for i in range(num_agents)]
        self.critics = [DDPGCritic(i, state_size, action_size, lr_critic, tau, batch_size, hidden_units) for i in range(num_agents)]
        
        self.noise = OUNoise(num_agents, action_size, self.seed, self.epsilon)
        
    def set_session(self, sess):
        self.sess = sess
        
    def reset(self):
        self.noise.reset(self.epsilon)
        self.critic_loss = []
        
    def step(self, state, action, reward, next_state, done):        
        for i in range(self.num_agents):
            self.memories[i].add((state[i].reshape([-1]), action[i].reshape([-1]), reward[i], next_state[i].reshape([-1]), done[i]))
        
        if len(self.memories[0].buffer) >= self.batch_size and len(self.memories[1].buffer) >= self.batch_size:                
            self.learn()
        
    def act(self, states, isNoise):   
        actions0 = self.sess.run(self.actors[0].actions, feed_dict={self.actors[0].states: states[0].reshape(1, self.state_size)})
        actions1 = self.sess.run(self.actors[1].actions, feed_dict={self.actors[1].states: states[1].reshape(1, self.state_size)})
        actions = np.vstack([actions0, actions1])

        if isNoise:
            actions += self.noise.sample()        
            
        return np.clip(actions, -1., 1.)
        
    def learn(self):
        for i in range(self.num_agents):
            experiences = self.memories[i].sample(self.batch_size)
            states, actions, rewards, next_states, dones = self.memories[i].unpack(experiences)
            
            next_actions = self.sess.run(self.actors[i].target_actions, feed_dict={self.actors[i].target_states: next_states})
            target_Qs = self.sess.run(self.critics[i].target_Qs, feed_dict={self.critics[i].target_states: next_states, \
                                                                        self.critics[i].target_actions: next_actions})

            rewards = np.array(rewards).reshape((self.batch_size, 1))
            dones = np.array(dones).reshape((self.batch_size, 1))
            Ys = rewards + (1 - dones) * self.gamma * target_Qs

            loss, _ = self.sess.run([self.critics[i].loss, self.critics[i].opt], feed_dict={self.critics[i].Ys: Ys, \
                                                                               self.critics[i].states: states, \
                                                                               self.critics[i].actions: actions})

            predicted_actions = self.sess.run(self.actors[i].actions, feed_dict={self.actors[i].states: states})

            critic_grads = self.sess.run(self.critics[i].grads, feed_dict={self.critics[i].states: states, \
                                                                 self.critics[i].actions: predicted_actions})   

            _ = self.sess.run(self.actors[i].opt, feed_dict={self.actors[i].states: states, self.actors[i].critic_grads: critic_grads})
            
        self.critic_loss.append(loss)        
        self.soft_update()        
        self.epsilon -= self.epsilon_decay
    
    def hard_update(self):
        for i in range(self.num_agents):
            self.sess.run([self.actors[i].hard_update, self.critics[i].hard_update])
    
    def soft_update(self):
        for i in range(self.num_agents):
            self.sess.run([self.actors[i].soft_update, self.critics[i].soft_update])                 