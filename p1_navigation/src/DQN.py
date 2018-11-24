import tensorflow as tf

class Network():
    def __init__(self, state_size, action_size, hidden_size, learning_rate, scope):        
        with tf.variable_scope(scope, reuse=False):        
            self.states = tf.placeholder(tf.float32, [None, state_size], name='states')
            self.Ys = tf.placeholder(tf.float32, [None, 1], name='targetQ')

            self.actions = tf.placeholder(tf.int32, [None], name='actions')
            self.one_hot_actions = tf.one_hot(self.actions, action_size)

            self.hidden1 = tf.layers.dense(self.states, hidden_size, tf.nn.relu, name='hidden1')
            self.hidden2 = tf.layers.dense(self.hidden1, hidden_size, tf.nn.relu, name='hidden2')

            self.hidden_V1 = tf.layers.dense(self.hidden2, hidden_size / 2, tf.nn.relu, name='hidden_V1')
            self.V = tf.layers.dense(self.hidden_V1, 1, None, name="state_function")            

            self.hidden_As1 = tf.layers.dense(self.hidden2, hidden_size / 2, tf.nn.relu, name='hidden_As1')            
            self.As = tf.layers.dense(self.hidden_As1, action_size, None, name="action_function")

            self.Qs = self.V + tf.subtract(self.As, tf.reduce_mean(self.As, axis=1, keepdims=True))

            self.Q = tf.reduce_sum(tf.multiply(self.Qs, self.one_hot_actions), axis=1, keepdims=True)

            self.loss = tf.reduce_mean(tf.square(self.Ys - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)        