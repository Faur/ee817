
# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50000#50000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
#ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
#CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
#TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = False
# Use Gym Monitor
GYM_MONITOR_EN = False  # nonworking on my machine
# Gym environment
ENV_NAME = 'GazeboCircuit2TurtlebotLidarNn-v0'
        #'CartPole-v0'
        #  'GazeboCircuit2TurtlebotLidarNn-v0' (state size:100)
        #'GazeboCircuit2TurtlebotLidar-v0'# (state size: 5, numbers in {0,1})
        # 'MountainCarContinuous-v0', 'Pendulum-v0'
ENV_NAME_SHORT = 'gazebo' #"cartpole"
ENV_IS_A_GAZEBO = True
if ENV_IS_A_GAZEBO:
    NR_ACTIONS = 21
    NR_OBSERVATIONS = 100

MODEL_NAME = "vanillaPG"
# Directory for storing gym results
MONITOR_DIR = './results/gym_monitor_' + MODEL_NAME + "_" +ENV_NAME_SHORT
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_'+ MODEL_NAME + "_" + ENV_NAME_SHORT

LOG = False
LOG_EVERY = 1000
# Whether to restore from existing weights:
RESTORE = False
STORE_WEIGHTS = True
# Directory to store weights to and load weights from
if STORE_WEIGHTS:
    WEIGHTS_DIR = "./weights/weights_"+MODEL_NAME + "_" + ENV_NAME_SHORT + "/"
    STORE_WEIGHTS_EVERY = 500
if RESTORE:
    RESTORE_EP = 0
    # WEIGHTS_TO_RESTORE = WEIGHTS_DIR + "ep_50_320.ckpt"
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 5 * 64

DEBUG = True

# ===============================
#   Algorithm-specific Parameters
# ===============================
# Store plain rewards, or a decayed sum of rewards?
# (In the second case, can only store experiences at episode end.)
RAW_R = False
# If on-policy method, usually don't want to use old experiences from before last update
USE_OLD_EXP = False