import numpy as np
import tensorflow as tf
# from tensorflow.python import debug as tf_debug
import matplotlib
matplotlib.use('Agg')
import gym
from gym import wrappers
import gym.spaces
import datetime

import gym_gazebo.envs.gazebo_circuit2_turtlebot_lidar_nn
import gym_gazebo

from ddpg_learner import DDPGLearner
from vanillaPG_learner import VanillaPGLearner
from summary_manager import *
from gym_utils import *
from replay_buffer import *
from pyplot_logging import *

#
#   Copy-paste this and adjust the constants in the beginning (capital letters) to
#       train a different model or environment.
#

## Not todo: ?saving and restarting from weights doesn't work yet. It pretends it does, but no progress is remembered.
##      Todo: -- Or it does? I think it does. Todo: test more thoroughly.

# ===========================
# * Choose your setting.
# ===========================

#setting = "ddpg_pendulum"
#from globals_ddpg_pendulum import *

#setting = "vanillaPG_cartpole"
#from globals_vanillaPG_cartpole import *

setting = "vanillaPG_gazebo"
from globals_vanillaPG_gazebo import *

#  todo
# setting = "ddpg_pendulum_from_pixels"
#setting = "DQN_cartpole"
#setting = "DQN_gazebo"



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
    elif ENV_IS_A_GAZEBO:
        env = wrappers.Monitor(env=env, directory=MONITOR_DIR, resume=False, force=True)

    if ENV_IS_A_GAZEBO:

        env.action_space = gym.spaces.Discrete(NR_ACTIONS)
        env.observation_space = gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(NR_OBSERVATIONS,))

    # ===========================
    # * Set up the learner
    if "ddpg" in setting:
        learner = DDPGLearner(observation_space=env.observation_space, action_space=env.action_space)
    elif "vanillaPG" in setting:
        learner = VanillaPGLearner(observation_space=env.observation_space, action_space=env.action_space)
    else:
        raise NotImplementedError


    #learner.sess = tf_debug.LocalCLIDebugWrapperSession(learner.sess)
    #learner.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    saver = tf.train.Saver()
    if RESTORE:
        weights_path = WEIGHTS_DIR + "ep_"+str(RESTORE_EP)+"_" + str(MINIBATCH_SIZE) +".ckpt"
        #saver.restore(learner.sess, WEIGHTS_TO_RESTORE)
        saver.restore(learner.sess, weights_path)
        #print("Model restored from file: %s" % WEIGHTS_TO_RESTORE)
        print("Model restored from file: %s" % weights_path)

    # ===========================
    # * Set up logging, if needed
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    tf_logdir = SUMMARY_DIR + "_" + str(MINIBATCH_SIZE) + "_tf/" + timestr
    logdir = SUMMARY_DIR + "_" + str(MINIBATCH_SIZE) + "/" + timestr

    # this still carries along q values; they are just 0 for if learner doesn't overwrite
    #   the function "filter_output_qvals()."
    log_variable_names_ep = ["av_min_q", "av_max_q","survival_time", "ep_reward", "train_count", "avg_reward"]
    log_variable_names_train = learner.other_training_stats_names
    log_variable_names_step = []
    for i in range(learner.s_dim()):
        log_variable_names_step.append("state_"+str(i))
    for i in range(learner.a_dim()):
        log_variable_names_step.append("action_"+str(i))
    for name in learner.other_prediction_stats_names:
        log_variable_names_step.append(name)

    tf_summarizer = SummaryManager(tf_logdir, learner.sess, graph=learner.sess.graph, variable_names_ep=log_variable_names_ep,
                                                                        variable_names_step=log_variable_names_step)
    logger = Logger(logdir)

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    # ===========================
    # * Start training
    # ===========================

    try:
        train_count = 0
        ep_rewards = []

        start_ep = RESTORE_EP if RESTORE else 0

        for ep in range(start_ep + 1, MAX_EPISODES):

            ep_reward = 0
            av_q_max_ep = 0
            av_q_min_ep = 0
            ep_history = []
            s = env.reset()



            for j in range(MAX_EP_STEPS):

                a, other_pred_stats = learner.get_action(s, ep)

                s2, r, terminated, _ = env.step(a)

                if RENDER_ENV:
                    s2_img = env.render()


                #  Add experience to buffer
                if RAW_R:
                    ## Todo: this might throw errors for spaces with more than one dimension -?
                    replay_buffer.add(np.reshape(s, (learner.s_dim(),)), np.reshape(a, (learner.a_dim(),)), r,
                                      terminated, np.reshape(s2, (learner.s_dim(),)))
                else:
                    ep_history.append([s, a, r, int(terminated), s2])   # we'll cast that list to a np.array, so better only have it contain numbers

                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                log_vals_step = []
                log_vals_ep = []
                log_vals_train = []
                prefilled_summaries_train = []
                prefilled_summaries_ep = []
                gradients = None
                if replay_buffer.size() > MINIBATCH_SIZE:

                    # Sample batch
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        replay_buffer.sample_batch(MINIBATCH_SIZE)
                    if not USE_OLD_EXP:
                        replay_buffer.clear()

                    #  Train
                    gradients, summaries, losses, loss_summaries, other_training_stats \
                        = learner.train(s_batch, a_batch, r_batch, t_batch, s2_batch)
                    train_count += 1

                    if DEBUG:
                        if np.isnan(np.max([np.max(g) for g in gradients])):
                            print("waah")
                        print('| Grad-mean: %.2f' % (np.mean([np.mean(np.abs(g))for g in gradients]) ),
                              '| Grad-max: %.2f' % np.max([np.max(np.abs(g)) for g in gradients]))
                    # If the learner doesn't overwrite that function, these
                    #   are just zero
                    q_vals = learner.filter_output_qvals(other_training_stats, other_pred_stats)
                    av_q_max_ep += max(q_vals)
                    av_q_min_ep += min(q_vals)

                    log_vals_train += other_training_stats

                    prefilled_summaries_train = summaries + loss_summaries

                    # collect gradients
                    logger.collect(dict(zip(learner.gradient_names, np.absolute(gradients))))
                    logger.collect(dict(zip(learner.loss_names, losses)))

                    # collect any other training variables
                    logger.collect(dict(zip(log_variable_names_train, log_vals_train)))

                #  Log state, action etc every step
                if learner.s_dim() == 1:
                    log_vals_step = [s]
                elif learner.s_dim() > 1:
                    log_vals_step = [s[k] for k in range(learner.s_dim())]
                if learner.a_dim() == 1:
                    log_vals_step.append(a)
                elif learner.a_dim() > 1:
                    log_vals_step += [a[k] for k in range(learner.a_dim())]
                log_vals_step += other_pred_stats

                if LOG and ep % LOG_EVERY == 0:
                    logger.track(dict(zip(log_variable_names_step, log_vals_step)), step_nr=j)

                s = s2
                ep_reward += r


                if terminated:
                    ep_rewards.append(ep_reward)            #  ["min_q_val", "max_q_val","survival_time", "ep_reward"]
                    log_vals_ep = [av_q_min_ep / float(j), av_q_max_ep / float(j), j, ep_reward, train_count, np.mean(ep_rewards)]

                tf_summarizer.log(learner.sess, ep, j, var_vals_ep=log_vals_ep, var_vals_step=log_vals_step,terminated=terminated,
                                   prefilled_summaries_ep=prefilled_summaries_ep, prefilled_summaries_step=prefilled_summaries_train)

                if terminated:

                        print('| Reward: %.2i' % int(ep_reward), " | Episode", ep,
                              '| Qmax: %.4f' % (av_q_max_ep / float(j)))
                        if not gradients is None:
                            print('| Grad-mean: %.2f' % (np.mean([np.mean(np.abs(g)) for g in gradients])),
                                  '| Grad-max: %.2f' % np.max([np.max(np.abs(g)) for g in gradients]))

                        # add decayed rewards only once per episode
                        if not RAW_R:
                            ep_history = np.array(ep_history)
                            ep_history[:, 2] = discount_rewards(ep_history[:, 2], GAMMA)
                            for [s, a, r, term, s2] in ep_history:
                                replay_buffer.add(np.reshape(s, (learner.s_dim(),)), np.reshape(a, (learner.a_dim(),)),
                                                  r, term, np.reshape(s2, (learner.s_dim(),)))

                        # store plots for past episode
                        if LOG and ep % LOG_EVERY == 0 and log_vals_step != []:
                            logger.store_tracked(log_variable_names_step, title="ep-"+str(ep), step_name="steps")
                        # track per-episode variables
                        logger.track(dict(zip(log_variable_names_ep, log_vals_ep)), ep)

                        # track gradient statistics, if any training was done in this episode
                        #if not gradients is None:
                        if all([key in logger.dict_stats_collecting for key in learner.gradient_names]):
                            logger.merge_collected(learner.gradient_names, ep)
                            logger.merge_collected(learner.loss_names, ep)
                            logger.merge_collected(log_variable_names_train, ep)

                        break

            #
            if ENV_IS_A_GAZEBO:
                env._flush(force=True)

            if ep != 0 and ep % STORE_WEIGHTS_EVERY == 0:
                if not os.path.exists(WEIGHTS_DIR):
                    os.makedirs(WEIGHTS_DIR)
                saver.save(learner.sess, WEIGHTS_DIR + "ep_"+str(ep)+"_" + str(MINIBATCH_SIZE) +".ckpt")


    finally:

        # After training finished, store plots of all per-episode variables & stats
        try:
            logger.store_all("End-ep-"+str(ep), ep, step_name="episode")

        finally:

            if STORE_WEIGHTS:
                if not os.path.exists(WEIGHTS_DIR):
                    os.makedirs(WEIGHTS_DIR)
                try:
                    saver.save(learner.sess, WEIGHTS_DIR + "ep_" + str(ep) + "_" + str(MINIBATCH_SIZE) +".ckpt")
                    print "Stored final weights in: "+WEIGHTS_DIR + "ep_" + str(ep) + "_" + str(MINIBATCH_SIZE) + ".ckpt"
                except:
                    saver.save(learner.sess, WEIGHTS_DIR + "ep_unknown_" + "_" + str(MINIBATCH_SIZE) + ".ckpt")
                    print "Stored final weights in: " + WEIGHTS_DIR + "ep_unknown_" + "_" + str(MINIBATCH_SIZE) + ".ckpt"


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