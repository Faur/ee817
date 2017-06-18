import numpy as np
import gym
from gym import wrappers


#
#  Not completed yet
#



class DiscreteSwingup():

    def __init__(self, monitor_dir, render=False,  nr_bins=10):

        assert nr_bins > 1  # Otherwise there's no decision!

        self.env = gym.make('Pendulum-v0')

        #if not render:
        #    self.env = wrappers.Monitor(
        #        self.env, monitor_dir, video_callable=False, force=True)
        #else:
        #    self.env = wrappers.Monitor(self.env, monitor_dir, force=True)
        if not render:
            self.env = wrappers.Monitor(env=self.env, directory=monitor_dir, video_callable=False,  resume=False, force=True)
        else:
            self.env = wrappers.Monitor(env=self.env, directory=monitor_dir, resume=False, force=True)

       # self.monitor = self.env.monitor

        self.action_space = gym.spaces.Discrete(nr_bins)
        self.observation_space = self.env.observation_space

        self.nr_bins = nr_bins
        self.action_range = float(self.env.action_space.high - self.env.action_space.low)


    def make_action_continuous(self, a):
        # c_action is an array of shape (1):
        c_action = self.env.action_space.low + float(a) / float(self.nr_bins - 1.) * self.action_range
        assert c_action >= self.env.action_space.low[0] and c_action <= self.env.action_space.high[0] # you never know
        return c_action


    def reset(self):
        return self.env.reset()

    def step(self, action):
        action = self.make_action_continuous(action)
        return self.env.step(action)


    def seed(self, rand_seed):
        return self.env.seed(rand_seed)

    def _flush(self, force):
        return self.env._flush(force=force)

    def render(self):
        return self.env.render()