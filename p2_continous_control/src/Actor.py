import tensorflow as tf

class DDPGActor():
    def __init__(self, state_size, action_size, lr, tau, batch_size, hidden_units):
        self.lr = lr
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.batch_size = batch_size
        
        self.states = tf.placeholder(tf.float32, [None, self.state_size], name='states')
        self.target_states = tf.placeholder(tf.float32, [None, self.state_size], name='target_states')
        
        with tf.variable_scope('Actor'):            
            self.actions = self.build_net(self.states, hidden_units, 'Eval', True)
            self.target_actions = self.build_net(self.target_states, hidden_units, 'Target', False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Actor/Eval')            
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Actor/Target')    
        
        self.critic_grads = tf.placeholder(tf.float32, [None, self.action_size], name="critic_grads")            
        
        opt = tf.train.AdamOptimizer(self.lr)        
        self.grads_and_vars = opt.compute_gradients(self.actions, self.e_params, grad_loss=-self.critic_grads)             
        self.clipped_grads_and_vars = [(g / self.batch_size, v) for g, v in self.grads_and_vars]
        self.opt = opt.apply_gradients(self.clipped_grads_and_vars)
        
        self.no_critic = opt.compute_gradients(self.actions, self.e_params)
        self.no_clipped = [(g, v) for g, v in self.grads_and_vars]
        
        self.soft_update = [tf.assign(t, (1-tau) * t + tau * e) for t, e in zip(self.t_params, self.e_params)]
        self.hard_update = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
    
    def build_net(self, states, hidden_units, scope, trainable):
        with tf.variable_scope(scope):
            h1 = tf.layers.dense(states, hidden_units[0], tf.nn.relu, trainable=trainable, name='h1')
            h2 = tf.layers.dense(h1, hidden_units[1], tf.nn.relu, trainable=trainable, name='h2')
            actions = tf.layers.dense(h2, self.action_size, tf.nn.tanh, kernel_initializer=tf.initializers.random_uniform(-3e-3, -3e-3), trainable=trainable, name='actions')
        return actions