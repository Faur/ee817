import numpy as np
import gym
import gym.spaces



def get_gym_space_size(spc):

    if spc is None:
        return 0
    if type(spc) == gym.spaces.box.Box:
        return spc.shape[0]
    elif type(spc) == gym.spaces.Discrete:
        return 1
    elif type(spc) == gym.spaces.MultiDiscrete:
        return spc.shape
    else:
        raise NotImplementedError('Still need to implement size-getter for gym space class '+str(type(spc)))
        return -1
