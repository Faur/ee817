
# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 500#50000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Store plain rewards, or a decayed sum of rewards?
# (In the second case, can only store experiences at episode end.)
RAW_R = True
# Soft target update param
TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = False
# Use Gym Monitor
GYM_MONITOR_EN = False  # nonworking on my machine
# Gym environment
ENV_NAME = 'Pendulum-v0'  #'GazeboCircuit2TurtlebotLidar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0'
ENV_NAME_SHORT = "pendulum"
ENV_IS_A_GAZEBO = False
if ENV_IS_A_GAZEBO:
    NR_ACTIONS = 21
    NR_OBSERVATIONS = 100
LOG_EVERY = 100
# Whether to restore from existing weights:
MODEL_NAME = "ddpg"
# Directory for storing gym results
MONITOR_DIR = './results/gym_monitor_' + MODEL_NAME + "_" +ENV_NAME_SHORT
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/still_testing/tf_'+ MODEL_NAME + "_" + ENV_NAME_SHORT

RESTORE = True
STORE_WEIGHTS = True
# Directory to store weights to and load weights from
if STORE_WEIGHTS:
    WEIGHTS_DIR = "./weights/still_testing/weights_"+MODEL_NAME + "_" + ENV_NAME_SHORT + "/"
    STORE_WEIGHTS_EVERY = 100
if RESTORE:
    WEIGHTS_TO_RESTORE = WEIGHTS_DIR + "ep_2.ckpt"
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64
