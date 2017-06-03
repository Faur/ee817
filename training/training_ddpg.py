import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
import gym.spaces
import datetime

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

## Todo: saving and restarting from weights doesn't work yet. It pretends it does, but no progress is remembered.


# ===========================
# * Choose your setting.
# ===========================

#setting = "ddpg_pendulum"
#from globals_ddpg_pendulum import *

# setting = "vanillaPG_cartpole"
# from globals_vanillaPG_cartpole import *

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


    saver = tf.train.Saver()
    if RESTORE:
        saver.restore(learner.sess, WEIGHTS_TO_RESTORE)
        print("Model restored from file: %s" % WEIGHTS_TO_RESTORE)

    # ===========================
    # * Set up logging, if needed
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    tf_logdir = SUMMARY_DIR + "_tf/" + timestr
    logdir = SUMMARY_DIR + "/" + timestr

    # this still carries along q values; they are just 0 for if learner doesn't overwrite
    #   the function "filter_output_qvals()."
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
                var_vals_step = []
                var_vals_ep = []
                prefilled_summaries_step = []
                prefilled_summaries_ep = []
                gradients = None
                if replay_buffer.size() > MINIBATCH_SIZE:

                    # Sample batch
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        replay_buffer.sample_batch(MINIBATCH_SIZE)

                    #  Train
                    gradients, summaries, losses, loss_summaries, other_training_stats \
                        = learner.train(s_batch, a_batch, r_batch, t_batch, s2_batch)

                    # If the learner doesn't overwrite that function, these
                    #   are just zero
                    q_vals = learner.filter_output_qvals(other_training_stats, other_pred_stats)
                    av_q_max_ep += max(q_vals)
                    av_q_min_ep += min(q_vals)

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
                    logger.collect(dict(zip(learner.gradient_names, np.absolute(gradients))))
                    logger.collect(dict(zip(learner.loss_names, losses)))

                    # log per-step variables
                    if ep % LOG_EVERY == 0:
                        logger.track(dict(zip(log_variable_names_step, var_vals_step)), step_nr=j)

                s = s2
                ep_reward += r


                if terminated:   #  ["min_q_val", "max_q_val","survival_time", "ep_reward"]
                    var_vals_ep = [av_q_min_ep / float(j), av_q_max_ep / float(j), j, ep_reward]

                tf_summarizer.log(learner.sess, ep, j, var_vals_ep=var_vals_ep, var_vals_step=var_vals_step,terminated=terminated,
                                   prefilled_summaries_ep=prefilled_summaries_ep, prefilled_summaries_step=prefilled_summaries_step)

                if terminated:

                        print('| Reward: %.2i' % int(ep_reward), " | Episode", ep, \
                              '| Qmax: %.4f' % (av_q_max_ep / float(j)))

                        # add decayed rewards only once per episode
                        if not RAW_R:
                            ep_history = np.array(ep_history)
                            ep_history[:, 2] = discount_rewards(ep_history[:, 2], GAMMA)
                            for [s, a, r, term, s2] in ep_history:
                                replay_buffer.add(np.reshape(s, (learner.s_dim(),)), np.reshape(a, (learner.a_dim(),)),
                                                  r, term, np.reshape(s2, (learner.s_dim(),)))

                        # store plots for past episode
                        if ep % LOG_EVERY == 0 and var_vals_step != []:
                            logger.store_tracked(log_variable_names_step, title="just_testing_"+"ep-"+str(ep), step_name="steps")
                        # track per-episode variables
                        logger.track(dict(zip(log_variable_names_ep, var_vals_ep)), ep)
                        # track gradient statistics
                        if not gradients is None:
                            logger.merge_collected(learner.gradient_names, ep)
                            logger.merge_collected(learner.loss_names, ep)

                        break

            #
            if ENV_IS_A_GAZEBO:
                env._flush(force=True)

            if ep != 0 and ep % STORE_WEIGHTS_EVERY == 0:
                if not os.path.exists(WEIGHTS_DIR):
                    os.makedirs(WEIGHTS_DIR)
                saver.save(learner.sess, WEIGHTS_DIR + "ep_"+str(ep)+".ckpt")


    finally:

        # After training finished, store plots of all per-episode variables & stats
        logger.store_all("just_testing", ep, step_name="episode")

        if STORE_WEIGHTS:
            if not os.path.exists(WEIGHTS_DIR):
                os.makedirs(WEIGHTS_DIR)
            try:
                saver.save(learner.sess, WEIGHTS_DIR + "ep_" + str(ep) + ".ckpt")
                print "Stored final weights in: "+WEIGHTS_DIR + "ep_" + str(ep) + ".ckpt"
            except:
                saver.save(learner.sess, WEIGHTS_DIR + "ep_unknown.ckpt")
                print "Stored final weights in: " + WEIGHTS_DIR + "ep_unknown.ckpt"


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