import numpy as np
import tensorflow as tf
import math

LEARNING_RATE = 0.0001
BATCH_SIZE = 64
TAU = 0.001


class TargetNet:
    """ Target Network of RND """

    def __init__(self, num_states):
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()

            # target network model parameters:
            self.W1_a, self.B1_a, self.W2_a, self.B2_a, self.W3_a, self.B3_a,\
                self.state_in, self.target_network = self.create_target_net(
                    num_states)

            self.sess.run(tf.global_variables_initializer())

    def create_target_net(self, num_states):
        """ Network that takes states and return feature """
        N_HIDDEN_1 = 400
        N_HIDDEN_2 = 300
        state_in = tf.placeholder("float", [None, num_states])
        W1_a = tf.Variable(tf.random_uniform(
            [num_states, N_HIDDEN_1], -1 / math.sqrt(num_states), 1 / math.sqrt(num_states)))
        B1_a = tf.Variable(tf.random_uniform(
            [N_HIDDEN_1], -1 / math.sqrt(num_states), 1 / math.sqrt(num_states)))
        W2_a = tf.Variable(tf.random_uniform(
            [N_HIDDEN_1, N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1), 1 / math.sqrt(N_HIDDEN_1)))
        B2_a = tf.Variable(tf.random_uniform(
            [N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1), 1 / math.sqrt(N_HIDDEN_1)))
        W3_a = tf.Variable(tf.random_uniform([N_HIDDEN_2, 1], -0.003, 0.003))
        B3_a = tf.Variable(tf.random_uniform([1], -0.003, 0.003))

        H1_a = tf.nn.softplus(tf.matmul(state_in, W1_a) + B1_a)
        H2_a = tf.nn.tanh(tf.matmul(H1_a, W2_a) + B2_a)
        target_net = tf.matmul(H2_a, W3_a) + B3_a
        return W1_a, B1_a, W2_a, B2_a, W3_a, B3_a, state_in, target_net

    def target_net(self, state_t):
        return self.sess.run(self.target_network, feed_dict={self.state_in: state_t})




