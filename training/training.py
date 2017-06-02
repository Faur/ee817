import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
import datetime

import ddpg_learner
from summary_manager import *
from gym_utils import *
from replay_buffer import *
from pyplot_logging import *



# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 100#50000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
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
ENV_IS_A_GAZEBO = False
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/still_testing/tf_ddpg_pendulum'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64


# Todo: add additionals returned by learner to logging
# Todo: actually log the summaries
# Todo: write a logger making actual plots


def main():
    # ===========================
    # * Set up the environment

    env = gym.make(ENV_NAME)
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)
    if GYM_MONITOR_EN:
        if not RENDER_ENV:
            env = wrappers.Monitor(
                env, MONITOR_DIR, video_callable=False, force=True)
        else:
            env = wrappers.Monitor(env, MONITOR_DIR, force=True)


    # ===========================
    # * Set up the learner

    learner = ddpg_learner.DDPGLearner(observation_space=env.observation_space, action_space=env.action_space)

    # ===========================
    # * Set up logging, if needed
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    tf_logdir = SUMMARY_DIR + "_tf/" + timestr
    logdir = SUMMARY_DIR + "/" + timestr
    #writer = tf.summary.FileWriter(SUMMARY_DIR + "/" + timestr, learner.get_graph())

    # for DDPG here:
    log_variable_names_ep = ["av_min_q", "av_max_q","survival_time", "ep_reward"]
    log_variable_names_step = []
    for i in range(learner.s_dim()):
        log_variable_names_step.append("state_"+str(i))
    for i in range(learner.a_dim()):
        log_variable_names_step.append("action_"+str(i))

    for name in learner.other_prediction_stats_names:
        log_variable_names_step.append(name)
    for name in  learner.other_training_stats_names:
        log_variable_names_step.append("training_"+name)

    tf_summarizer = SummaryManager(tf_logdir, learner.sess, graph=learner.sess.graph, variable_names_ep=log_variable_names_ep,
                                                                        variable_names_step=log_variable_names_step)
    logger = Logger(logdir)

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    # ===========================
    # * Start training
    # ===========================

    try:
        for ep in range(MAX_EPISODES):

            ep_reward = 0
            av_q_max_ep = 0
            av_q_min_ep = 0
            s = env.reset()

            for j in range(MAX_EP_STEPS):

                a, other_pred_stats = learner.get_action(s, ep)

                s2, r, terminated, _ = env.step(a)

                if RENDER_ENV:
                    s2_img = env.render()


                #  Add experience to buffer
                ## Todo: this might throw errors for spaces with more than one dimension -?
                replay_buffer.add(np.reshape(s, (learner.s_dim(),)), np.reshape(a, (learner.a_dim(),)), r,
                                  terminated, np.reshape(s2, (learner.s_dim(),)))

                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                var_vals_step = []
                var_vals_ep = []
                prefilled_summaries_step = []
                prefilled_summaries_ep = []
                gradients = None
                if replay_buffer.size() > MINIBATCH_SIZE:
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        replay_buffer.sample_batch(MINIBATCH_SIZE)

                    #  Train
                    gradients, summaries, losses, loss_summaries, other_training_stats \
                        = learner.train(s_batch, a_batch, r_batch, t_batch, s2_batch)

                    #  Log gradients


                    av_q_max_ep += max(other_training_stats[0])
                    av_q_min_ep += min(other_training_stats[0])

                    other_training_stats = [np.mean(otherstat) for otherstat in other_training_stats]
                    other_pred_stats = [np.mean(otherstat) for otherstat in other_pred_stats]


                    #  Log state, action etc
                    if learner.s_dim() == 1:
                        var_vals_step = [s]
                    elif learner.s_dim() > 1:
                        var_vals_step = [s[k] for k in range(learner.s_dim())]
                    if learner.a_dim() == 1:
                        var_vals_step.append(a)
                    elif learner.a_dim() > 1:
                        var_vals_step += [a[k] for k in range(learner.a_dim())]
                    var_vals_step += other_pred_stats
                    var_vals_step += other_training_stats

                    prefilled_summaries_step = summaries + loss_summaries

                    # collect gradients
                    logger.collect(dict(zip(learner.gradient_names, gradients)))



                s = s2
                ep_reward += r


                if terminated:   #  ["min_q_val", "max_q_val","survival_time", "ep_reward"]
                    var_vals_ep = [av_q_min_ep / float(j), av_q_max_ep / float(j), j, ep_reward]

                tf_summarizer.log(learner.sess, ep, j, var_vals_ep=var_vals_ep, var_vals_step=var_vals_step,terminated=terminated,
                                   prefilled_summaries_ep=prefilled_summaries_ep, prefilled_summaries_step=prefilled_summaries_step)
                # log per-step variables
                logger.track(dict(zip(log_variable_names_step, var_vals_step)), step_nr=j)

                if terminated:
                    print('| Reward: %.2i' % int(ep_reward), " | Episode", ep, \
                          '| Qmax: %.4f' % (av_q_max_ep / float(j)))

                    # store plots for past episode
                    logger.store_tracked(log_variable_names_step, title="just_testing_"+"ep-"+str(ep), step_name="steps")
                    # track per-episode variables
                    logger.track(dict(zip(log_variable_names_ep, var_vals_ep)), ep)
                    # track gradient statistics
                    if not gradients is None:
                        logger.merge_collected(learner.gradient_names, ep)

                    break

            if ENV_IS_A_GAZEBO:
                env._flush(force=True)

    finally:


        # After training finished, store plots of all per-episode variables & stats
        logger.store_all("just_testing", ep, step_name="episode")




        # ===========================
        # * Shut down
        # ===========================
        if GYM_MONITOR_EN:
            env.monitor.close()

        if ENV_IS_A_GAZEBO:
            from subprocess import call
            call(["killall -9 gzserver gzclient roslaunch rosmaster"], shell=True)



if __name__ == '__main__':
    #main()
    main()