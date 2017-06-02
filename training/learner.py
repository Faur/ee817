
import numpy as np
import tensorflow as tf
import gym

from gym_utils import *


#
# To inherit from this class, overwrite "train()" and "get_action()", as
#  well as __init__(). In the latter, please set self.activation_space and
#  self.observation_space. You can just pass these two as variables to __init__().
#
# It's easiest, if that's possible, if a child class keeps the same arguments to
#  train() and get_action(), and the same return values. Otherwise this class isn't
#  much use ;)
#


class Learner():

    def __init__(self):


        self.gradient_names = []    # should have same shape as
        #self.summaries = []
        self.loss_names = []
        self.other_training_stats_names = []
        self.other_prediction_stats_names = []

        self.action_space = None    # maybe use openAI-gym "gym.box.Box"/"...Discrete" variables here
        self.observation_space = None     # "Box"/"Discrete"; if image input: use "Box"



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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

