
import numpy as np
import tensorflow as tf
import gym

from gym_utils import *


#
# To inherit from this class, overwrite "train()" and "get_action()", as
#  well as __init__(). In the constructor, please set the variables mentioned.
#  self.activation_space and self.observation_space you can just expect to be passed
#  as variables to __init__().
#  For the other variables:
#            self.gradient_names:
#           A list with gradient names, same length as the list "gradients"/first result returned by learner.train().

#            self.loss_names = []
#           A list with loss names, same length as the list "losses"/third result returned by learner.train().

#            self.other_training_stats_names = []
#           A list with same length as the last return value of learner.train().

#            self.other_prediction_stats_names = []
#           A list of same length as the last return value of learner.get_action().
#
#           Please also call:
#            self.sess = tf.Session()  # learner is expected to own a session
#            self.sess.run(tf.global_variables_initializer())
#
#
#
# Please keep the same arguments to
#  train() and get_action(), and the same return values. Otherwise this class is not
#  much use ;)
#
# ! And: !  (other_training_stats etc:)
# ********
# * The return values "other_training_stats" and "other_prediction_stats" are expected
# *  to be a list each (not arrays). Each list entry can be either a scalar, or a list,
# *  or a numpy array. If it is a list, or a numpy array with *one* dimension (i.e.,
# *  len(a.shape)==1), it is interpreted as values for a list of variables that will be
# *  plotted together in one plot.
# *  If it's a scalar, it will become a one-line plot.
# *  If it's a bigger array, logging will calculate average and percentiles of these
# *  values. (> pyplot_logging.py)
#
#
#

class Learner():
    '''
    self.gradient_names = []    # should have same shape as
        #self.summaries = []
        self.loss_names = []
        self.other_training_stats_names = []
        self.other_prediction_stats_names = []

        self.action_space = None    # maybe use openAI-gym "gym.box.Box"/"...Discrete" variables here
        self.observation_space = None     # "Box"/"Discrete"; if image input: use "Box"

        self.sess = tf.Session() # learner is expected to own a session
        self.sess.run(tf.global_variables_initializer())

    '''

    def __init__(self, a_space=None, o_space=None):


        self.gradient_names = []
        self.loss_names = []
        self.other_training_stats_names = []
        self.other_prediction_stats_names = []

        self.action_space = a_space    # maybe use openAI-gym "gym.box.Box"/"...Discrete" variables here
        self.observation_space = o_space     # "Box"/"Discrete"; if image input: use "Box"

        self.sess = tf.Session() # learner is expected to own a session
        self.sess.run(tf.global_variables_initializer())


    def train(self, s_batch, a_batch, r_batch, t_batch, s2_batch):
        ''' Update the learner's parameters according to the training input.
        :param s_batch: a batch of states, should fit self.input_space
        :param a_batch: acctions taken, at each state in s_batch
        :param r_batch: corresponding rewards - depending on the learner, pure rewards, or accumulated rewards
        :param t_batch: whether the environment terminated after this step
        :param s2_batch: state we ended up in
        :return: gradients: all gradients' values during this update; list of numpy-matrices
        :return: summaries: list with any summary strings that were created during the update
        :return: other_training_stats
        '''
        raise NotImplementedError("Please implement a training function for this learner.")
        return gradients, summaries, losses, loss_summaries, other_training_stats



    def get_action(self, s, t=1):
        '''
        Output an advised action.
        :param s: the state, should fit self.input_space
        :param t: optional timestep, for modulating exploration noise or similar
        :return: action, should fit self.action_space
        :return: other_prediction_stats: an empty list, or a list with other net statistics observed in this action estimation
        '''
        raise NotImplementedError("Please implement action prediction.")
        return action, other_prediction_stats



    # action size
    def a_dim(self):
        return get_gym_space_size(self.action_space)

    # observation size
    def s_dim(self):
        return get_gym_space_size(self.observation_space)

    # Difference to a_dim(): in the discrete case, return the number of discrete values.
    # Better size calculation for Box-type action spaces.
    def get_action_count(self):
        if self.action_space is None:
            return 0
        elif isinstance(self.action_space, gym.spaces.Discrete):
            return self.action_space.n
        elif isinstance(self.action_space, gym.spaces.Box):
            n = 1.
            for i in range(self.action_space.shape):
                n *= self.action_space.shape[i]
            return n
        else:
            raise NotImplementedError

    def filter_output_qvals(self, other_training_stats, other_prediction_stats):
        return np.array([0])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

