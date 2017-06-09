import numpy as np
import gym
from gym import wrappers





class ConinuifiedGazebo():
    '''
    Wrap the env to pretend it has a continuous action space, in one dimension.
    Also optionally reduce the observation space.
    '''
    def __init__(self, num_actions, num_obs, env_name='GazeboCircuit2TurtlebotLidarNn-v0',
                 render=False, gym_monitor_en=False, monitor_dir="", red_fact=1, rand=1234):

        self.env = gym.make(env_name)

        self.obs_reduction_factor = red_fact
        assert type(self.obs_reduction_factor) == int
        self.n_disc_actions = num_actions
        self.fake_n_obs = np.floor(num_obs / self.obs_reduction_factor).astype(int)
        if self.fake_n_obs == 0:
            print("Error continuifying the environment "+env_name+", cannot reduce "+str(num_actions)+" actions "
                   "by a factor "+str(red_fact)+".\n")
            raise ValueError

        np.random.seed(rand)
        self.env.seed(rand)
        if gym_monitor_en:
            if not render:
                self.env = wrappers.Monitor(
                    self.env, monitor_dir, video_callable=False, force=True)
            else:
                self.env = wrappers.Monitor(self.env, monitor_dir, force=True)

        else:
            self.env = wrappers.Monitor(env=self.env, directory=monitor_dir, resume=False, force=True)

        self.env.action_space = gym.spaces.Discrete(num_actions)
        self.env.observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(num_obs,))

#        self.monitor = self.env.monitor
        #self.action_space = gym.spaces.Box(low=0, high=num_actions-1, shape=(1,))
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(1,))
        self.observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(self.fake_n_obs,))  #self.env.observation_space


    def reduce_state_dim(self, s):
        return s[::self.obs_reduction_factor]

    #def make_action_cont(self, a):
    #    a = (a + 1.) * (self.n_disc_actions - 1.)
    #    a = np.clip(np.round(a), 0, self.n_disc_actions - 1).astype(int)
    #    return a

    def reset(self):
        s  = self.env.reset()
        return self.reduce_state_dim(s)

    def step(self, a):
        a = (a + 1.) / 2. * (self.n_disc_actions - 1.)
        a = np.clip(np.round(a), 0, self.n_disc_actions - 1 ).astype(int)
        s2, r, terminated, info =  self.env.step(a)
        s2 = self.reduce_state_dim(s2)
        return s2, r, terminated, info




    def seed(self, rand_seed):
        return self.env.seed(rand_seed)

    def _flush(self, force):
        return self.env._flush(force=force)

    def render(self):
        return self.env.render()






# Todo: Implement this and run vanillaPG + gazebo again with this.
class ReducedInputGazebo():
    pass
