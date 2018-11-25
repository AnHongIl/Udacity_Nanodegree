import tensorflow as tf

class DDPGCritic():
    def __init__(self, index, state_size, action_size, lr, tau, batch_size, hidden_units):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.index = index
        
        self.states = tf.placeholder(tf.float32, [None, self.state_size], name='states')
        self.actions = tf.placeholder(tf.float32, [None, self.action_size], name='actions')
        
        self.target_states = tf.placeholder(tf.float32, [None, self.state_size], name='target_states')
        self.target_actions = tf.placeholder(tf.float32, [None, self.action_size], name='target_actions')
                   
        prefix = 'Critic' + str(index)
        with tf.variable_scope(prefix):            
            self.Qs = self.build_net(self.states, self.actions, hidden_units, 'Eval', True)
            self.target_Qs = self.build_net(self.target_states, self.target_actions, hidden_units, 'Target', False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, prefix + '/Eval')            
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, prefix + '/Target')    
        
        self.Ys = tf.placeholder(tf.float32, [None, 1], name='Ys')        
        
        self.loss = tf.reduce_mean(tf.square(self.Ys-self.Qs))

        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.grads = tf.gradients(self.Qs, self.actions)[0]

        self.soft_update = [tf.assign(t, (1-tau) * t + tau * e) for t, e in zip(self.t_params, self.e_params)]
        self.hard_update = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        
    def build_net(self, states, actions, hidden_units, scope, trainable):
        with tf.variable_scope(scope):
            h1 = tf.layers.dense(states, hidden_units[0], tf.nn.relu, trainable=trainable, name='h1')
            h2 = tf.layers.dense(tf.concat([h1, actions], axis=1), hidden_units[1], tf.nn.relu, trainable=trainable, name='h2')
            Qs = tf.layers.dense(h2, 1, None, kernel_initializer=tf.initializers.random_uniform(-3e-3, 3e-3), trainable=trainable, name='Qs')
        return Qs