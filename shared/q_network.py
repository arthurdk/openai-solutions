import tensorflow as tf
import numpy as np


class QNetwork(object):
    """
    Class representing an implementation of a Q-Learning method using Neural Networks
    """

    def __init__(self, in_dimension, out_dimension, discount_factor=0.99, start_epsilon=0.1,
                 decay_rate=0.91, decay_step=10, learning_rate=0.1):
        self.discount_factor = discount_factor
        self.start_epsilon = start_epsilon
        self.learning_rate = learning_rate

        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.cur_epsilon = start_epsilon

        # Tensorflow Objects
        self.Q_out = None
        self.prediction_op = None
        self.train_op = None
        self.next_Q = None
        self.states = None
        self.weights = None

    def create_network_graph(self):
        """
        Creates tensorflow computational graph
        :return:
        """
        tf.reset_default_graph()
        # These lines establish the feed-forward part of the network used to choose actions
        self.states = tf.placeholder(shape=[1, self.in_dimension], dtype=tf.float32)
        self.weights = tf.get_variable("weights", shape=[self.in_dimension, self.out_dimension],
                                       initializer=tf.contrib.layers.xavier_initializer())
        # Probability of each action
        self.Q_out = tf.matmul(self.states, self.weights)

        # We choose the one having the highest outcome
        self.prediction_op = tf.argmax(self.Q_out, 1)[0]

        self.next_Q = tf.placeholder(shape=[1, self.out_dimension], dtype=tf.float32)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        loss = tf.reduce_sum(tf.square(self.next_Q - self.Q_out))

        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
        # self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        return self.states, self.weights, self.Q_out, self.next_Q, self.train_op, self.prediction_op

    def get_current_epsilon(self):
        return self.cur_epsilon

    def end_episode(self, current_episode):
        # Reduce chance of random action as we train the model.
        # TODO change this by a proper decay
        """
        if self.cur_epsilon > self.min_epsilon and self.cur_epsilon - self.stepping_epsilon > self.min_epsilon:
            self.cur_epsilon -= self.stepping_epsilon
        """
        print((current_episode / self.decay_step))
        self.cur_epsilon *= self.decay_rate ** (current_episode / self.decay_step)

    def train(self, session, observation, targetQ):
        """
        Update model:
        Train the network after adding information corresponding to the observation
        :param session: Tensorflow session
        :param observation: gym observation
        :param targetQ: Score of each action available for this observation with updated value from the previously choosen action
        :return:
        """

        _, _ = session.run([self.train_op, self.weights],
                           feed_dict={self.states: np.identity(self.in_dimension)[observation:observation + 1],
                                      self.next_Q: targetQ})
