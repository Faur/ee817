import os


#
# Like this it's not really 'global variables' any more. Too lazy to find another name now.
#
class GlobalVars():

    def __init__(self):

        # ==========================
        #   Training Parameters
        # ==========================
        # Max training steps
        self.MAX_EPISODES = 50000
        # Max episode length
        self.MAX_EP_STEPS = 1000
        # Discount factor
        self.GAMMA = 0.99
        # Base learning rate for the Actor network, if any
        self.ACTOR_LEARNING_RATE = None  # p.emami: 0.0001
        # Base learning rate for the Critic Network, if any
        self.CRITIC_LEARNING_RATE = None  # p.emami: 0.001
        # Soft target update param
        self.TAU = None

        # ===========================
        #   Utility Parameters
        # ===========================
        # Render gym env during training
        self.RENDER_ENV = False
        self.RENDER_EVERY = 50
        # Use Gym Monitor
        self.GYM_MONITOR_EN = False  # nonworking on my machine
        # Gym environment
        self.ENV_NAME           = None #''GazeboCircuit2TurtlebotLidarNn-v0'  # 'GazeboCircuit2TurtlebotLidar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'
        self.ENV_NAME_SHORT     = None #"gazebo"
        self.MODEL_NAME         = None #""ddpg"
        self.NR_ACTIONS         = None
        self.NR_OBSERVATIONS    = None

        self.CONTINUIFY         = False
        self.ENV_IS_A_GAZEBO    = False
        self.CLIP_REWARDS       = False
        self.LOG                = True
        self.LOG_EVERY          = 100

        self.RESTORE = False
        self.RESTORE_EP = None
        self.STORE_WEIGHTS = False ## Doesn't work yet anyway
        # Directory to store weights to and load weights from
        self.STORE_WEIGHTS_EVERY = None
        # WEIGHTS_TO_RESTORE = WEIGHTS_DIR + "ep_20.ckpt"
        self.RANDOM_SEED = 1234
        # Size of replay buffer
        self.BUFFER_SIZE = 10000
        self.MINIBATCH_SIZE = 64
        self.LEARN_START = 0 # more precisely it is = max(LEARN_START, MINIBATCH_SIZE)

        self.DEBUG = False

        # ===============================
        #   Algorithm-specific Parameters
        # ===============================
        # Store plain rewards, or a decayed sum of rewards?
        # (In the second case, can only store experiences at episode end.)
        self.RAW_R = True
        # If on-policy method, usually don't want to use old experiences from before last update
        self.USE_OLD_EXP = True
        # For ddpg only:
        self.SIZE_HIDDEN = [300, 300]  # 300


    # Directory for storing gym results
    def MONITOR_DIR(self):
        return './results/gym_monitor_' + self.MODEL_NAME + "_" + self.ENV_NAME_SHORT

    # Directory for storing tensorboard summary results
    def SUMMARY_DIR(self):
        return './results/tf_' + self.MODEL_NAME + "_" + self.ENV_NAME_SHORT

    # for storing / loading weights
    def WEIGHTS_DIR(self):
        return "./weights/weights_" + self.MODEL_NAME + "_" + self.ENV_NAME_SHORT + "/"




    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    #   Settings
    #


    def set_ddpg_gazebo(self):
        # ==========================
        #   Training Parameters
        # ==========================
        # Max training steps
        self.MAX_EPISODES = 50000
        # Max episode length
        self.MAX_EP_STEPS = 1000
        # Discount factor
        self.GAMMA = 0.99
        # Base learning rate for the Actor network, if any
        self.ACTOR_LEARNING_RATE = 0.00005  # p.emami: 0.0001
        # Base learning rate for the Critic Network, if any
        self.CRITIC_LEARNING_RATE = 0.001  # p.emami: 0.001
        # Soft target update param
        self.TAU = 0.001

        # ===========================
        #   Utility Parameters
        # ===========================
        # Render gym env during training
        self.RENDER_ENV = False
        self.RENDER_EVERY = 50
        # Use Gym Monitor
        self.GYM_MONITOR_EN = False  # nonworking on my machine
        # Gym environment
        self.ENV_NAME           = 'GazeboCircuit2TurtlebotLidarNn-v0'  #''GazeboCircuit2TurtlebotLidarNn-v0'  # 'GazeboCircuit2TurtlebotLidar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'
        self.ENV_NAME_SHORT     = "gazebo" #"gazebo"
        self.MODEL_NAME         = "ddpg"
        self.NR_ACTIONS         = 21
        self.NR_OBSERVATIONS    = 100

        self.CONTINUIFY         = True
        self.ENV_IS_A_GAZEBO    = True
        self.CLIP_REWARDS       = True

        self.LOG                = True
        self.LOG_EVERY          = 100

        self.RESTORE = False
        self.RESTORE_EP = 3044
        self.STORE_WEIGHTS = False ## Doesn't work yet anyway
        # Directory to store weights to and load weights from
        self.STORE_WEIGHTS_EVERY = 500
        # WEIGHTS_TO_RESTORE = WEIGHTS_DIR + "ep_20.ckpt"
        self.RANDOM_SEED = 1234
        # Size of replay buffer
        self.BUFFER_SIZE = 100000 #10000
        self.MINIBATCH_SIZE = 64
        self.LEARN_START = 10000

        self.DEBUG = True

        # ===============================
        #   Algorithm-specific Parameters
        # ===============================
        # Store plain rewards, or a decayed sum of rewards?
        # (In the second case, can only store experiences at episode end.)
        self.RAW_R = True
        # If on-policy method, usually don't want to use old experiences from before last update
        self.USE_OLD_EXP = True
        # For ddpg only:
        self.SIZE_HIDDEN = [300, 300]  # 300

        return self






    def set_ddpg_pendulum(self):
        # ==========================
        #   Training Parameters
        # ==========================
        # Max training steps
        self.MAX_EPISODES = 1001
        # Max episode length
        self.MAX_EP_STEPS = 1000
        # Discount factor
        self.GAMMA = 0.99
        # Base learning rate for the Actor network, if any
        self.ACTOR_LEARNING_RATE = 0.0001  # p.emami: 0.0001
        # Base learning rate for the Critic Network, if any
        self.CRITIC_LEARNING_RATE = 0.001  # p.emami: 0.001
        # Soft target update param
        self.TAU = 0.00015 #0.001

        # ===========================
        #   Utility Parameters
        # ===========================
        # Render gym env during training
        self.RENDER_ENV = True
        self.RENDER_EVERY = 1
        # Use Gym Monitor
        self.GYM_MONITOR_EN = False  # nonworking on my machine
        # Gym environment
        self.ENV_NAME = 'Pendulum-v0'  # ''GazeboCircuit2TurtlebotLidarNn-v0'  # 'GazeboCircuit2TurtlebotLidar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'
        self.ENV_NAME_SHORT = "pendulum"  # "gazebo"
        self.MODEL_NAME = "ddpg"

        self.LOG = True
        self.LOG_EVERY = 100

        self.RESTORE = False
        self.RESTORE_EP = 500
        self.STORE_WEIGHTS = False  ## Doesn't work yet anyway
        # Directory to store weights to and load weights from
        self.STORE_WEIGHTS_EVERY = 500
        # WEIGHTS_TO_RESTORE = WEIGHTS_DIR + "ep_20.ckpt"
        self.RANDOM_SEED = 1234
        # Size of replay buffer
        self.BUFFER_SIZE = 10000
        self.MINIBATCH_SIZE = 64

        self.DEBUG = False

        # ===============================
        #   Algorithm-specific Parameters
        # ===============================
        # Store plain rewards, or a decayed sum of rewards?
        # (In the second case, can only store experiences at episode end.)
        self.RAW_R = True
        # If on-policy method, usually don't want to use old experiences from before last update
        self.USE_OLD_EXP = True
        # For ddpg only:
        self.SIZE_HIDDEN = [400, 300]  # 300

        return self







    def set_vanillaPG_gazebo(self):
        # ==========================
        #   Training Parameters
        # ==========================
        # Max training steps
        self.MAX_EPISODES = 50000
        # Max episode length
        self.MAX_EP_STEPS = 1000
        # Discount factor
        self.GAMMA = 0.99

        # ===========================
        #   Utility Parameters
        # ===========================
        # Render gym env during training
        self.RENDER_ENV = False
        self.RENDER_EVERY = 50
        # Use Gym Monitor
        self.GYM_MONITOR_EN = False  # nonworking on my machine
        # Gym environment
        self.ENV_NAME = 'GazeboCircuit2TurtlebotLidarNn-v0'  # ''GazeboCircuit2TurtlebotLidarNn-v0'  # 'GazeboCircuit2TurtlebotLidar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'
        self.ENV_NAME_SHORT = 'gazebo'  # "gazebo"
        self.CONTINUIFY = False
        self.ENV_IS_A_GAZEBO = True
        self.NR_ACTIONS = 21
        self.NR_OBSERVATIONS = 100
        self.LOG = True
        self.LOG_EVERY = 100
        # Whether to restore from existing weights:
        self.MODEL_NAME = "vanillaPG"
        # Directory for storing gym results

        self.RESTORE = False
        self.RESTORE_EP = 3044
        self.STORE_WEIGHTS = False  ## Doesn't work yet anyway
        self.STORE_WEIGHTS_EVERY = 500

        self.RANDOM_SEED = 1234
        # Size of replay buffer
        self.BUFFER_SIZE = 10000
        self.MINIBATCH_SIZE = 2 * 64

        self.DEBUG = False

        # ===============================
        #   Algorithm-specific Parameters
        # ===============================
        # Store plain rewards, or a decayed sum of rewards?
        # (In the second case, can only store experiences at episode end.)
        self.RAW_R = False
        # If on-policy method, usually don't want to use old experiences from before last update
        self.USE_OLD_EXP = False
        # For ddpg only:
        self.SIZE_HIDDEN = [300, 300]  # 300

        return self





    def set_vanillaPG_cartpole(self):
        # ==========================
        #   Training Parameters
        # ==========================
        # Max training steps
        self.MAX_EPISODES = 50000
        # Max episode length
        self.MAX_EP_STEPS = 1000
        # Discount factor
        self.GAMMA = 0.99


        # ===========================
        #   Utility Parameters
        # ===========================
        # Render gym env during training
        self.RENDER_ENV = False
        self.RENDER_EVERY = 50
        # Use Gym Monitor
        self.GYM_MONITOR_EN = False  # nonworking on my machine
        # Gym environment
        self.ENV_NAME           = 'CartPole-v0' #''GazeboCircuit2TurtlebotLidarNn-v0'  # 'GazeboCircuit2TurtlebotLidar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'
        self.ENV_NAME_SHORT     = "cartpole" #"gazebo"
        self.MODEL_NAME         = "vanillaPG" #""ddpg"

        self.LOG                = True
        self.LOG_EVERY          = 100

        self.RESTORE = False
        self.RESTORE_EP = 100
        self.STORE_WEIGHTS = True ## Doesn't work yet anyway
        # Directory to store weights to and load weights from
        self.STORE_WEIGHTS_EVERY = 100
        # WEIGHTS_TO_RESTORE = WEIGHTS_DIR + "ep_20.ckpt"
        self.RANDOM_SEED = 1234
        # Size of replay buffer
        self.BUFFER_SIZE = 10000
        self.MINIBATCH_SIZE = 64

        self.DEBUG = True

        # ===============================
        #   Algorithm-specific Parameters
        # ===============================
        # Store plain rewards, or a decayed sum of rewards?
        # (In the second case, can only store experiences at episode end.)
        self.RAW_R = False
        # If on-policy method, usually don't want to use old experiences from before last update
        self.USE_OLD_EXP = False
        # For ddpg only:
        self.SIZE_HIDDEN = [300, 300]  # 300

        return self


    def store_globals_txt(self, outfile):
        with open(outfile, "a+") as f:
            D = vars(self)
            for key in sorted(D.iterkeys()):
                val = D[key]
                f.write('%25s:  %s  \n' % (key, str(val)))
