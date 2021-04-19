import numpy as np
import tensorflow as tf
import math

LEARNING_RATE = 0.0001
BATCH_SIZE = 64
TAU = 0.001


class PredictNet:
    """ Predict Network of RND """

    def __init__(self, num_states):
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()

            # predictor network model parameters:
            self.W1_a, self.B1_a, self.W2_a, self.B2_a, self.W3_a, self.B3_a,\
                self.state_in, self.predict_network = self.create_predict_net(
                    num_states)


            self.predict_gradient_input = tf.placeholder("float", [None, 1])
            self.predict_parameters = [self.W1_a, self.B1_a,
                                     self.W2_a, self.B2_a, self.W3_a, self.B3_a]
            self.parameters_gradients = tf.gradients(
                self.predict_network, self.predict_parameters, -self.predict_gradient_input)  # /BATCH_SIZE)
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(
                zip(self.parameters_gradients, self.predict_parameters))
            # initialize all tensor variable parameters:
            self.sess.run(tf.global_variables_initializer())



    def create_predict_net(self, num_states):
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
        predict_net = tf.matmul(H2_a, W3_a) + B3_a
        return W1_a, B1_a, W2_a, B2_a, W3_a, B3_a, state_in, predict_net

    def predict_net(self, state_t):
        return self.sess.run(self.predict_network, feed_dict={self.state_in: state_t})


    def train_predict(self, state_in, q_gradient_input):
        i_state = np.zeros(shape=(64,3),dtype=np.int)

        for i in range(64):
            i_state[i][0] = state_in[i][0]
            i_state[i][1] = state_in[i][1]
            i_state[i][2] = state_in[i][2]
        self.sess.run(self.optimizer, feed_dict={
                      self.state_in: i_state, self.predict_gradient_input: q_gradient_input})


