import gym
import gym.spaces
import numpy as np
from gym import wrappers


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



def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r





def create_env(env_name, is_a_gazebo, render_env, monitor_dir, gym_monitor_en=False, rand_seed=9026,
               nr_actions=None, nr_obs=None):

    env = gym.make(env_name)
    np.random.seed(rand_seed)
    # tf.set_random_seed(rand_seed)
    env.seed(rand_seed)
    if gym_monitor_en:
        if not render_env:
            env = wrappers.Monitor(
                env, monitor_dir, video_callable=False, force=True)
        else:
            env = wrappers.Monitor(env, monitor_dir, force=True)
    elif is_a_gazebo:
        env = wrappers.Monitor(env=env, directory=monitor_dir, resume=False, force=True)
        assert None not in [nr_obs, nr_actions]
        env.action_space = gym.spaces.Discrete(nr_actions)
        env.observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(nr_obs,))


    return env
