# from tensorflow.python import debug as tf_debug
import matplotlib
matplotlib.use('Agg')
import datetime

from utilities import continuifier as cont

import gym
import gym_gazebo

#import gym_gazebo.envs.gazebo_circuit2_turtlebot_lidar_nn

from ddpg_learner_batchnormed import DDPGLearnerBN
from vanillaPG_learner import VanillaPGLearner
from utilities.summary_manager import *
from utilities.gym_utils import *
from utilities.replay_buffer import *
from utilities.pyplot_logging import *
from utilities.global_vars import *

# For memory monitoring:
#import memory_profiler
#from guppy import hpy
#h = hpy()

#
#   Copy-paste this and adjust the constants in the beginning (capital letters) to
#       train a different model or environment.
#

##      Todo: ?saving and restarting from weights doesn't work yet. It pretends it does, but no progress is remembered.
##      Todo: I think that's what the error message at storing / loading weights is for.

##      Todo: Test new way to set the global variables for other settings;
##      Todo:    still untested: vanillaPG+cartpole  -  ddpg+pendulum  -  ddpg+gazebo

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_up_logging(learner):
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    tf_logdir = g.SUMMARY_DIR() + "_" + str(g.MINIBATCH_SIZE) + "_tf/" + timestr
    logdir = g.SUMMARY_DIR() + "_" + str(g.MINIBATCH_SIZE) + "/" + timestr

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

    return tf_summarizer, logger, log_variable_names_step, log_variable_names_train, log_variable_names_ep


def list_state_and_action(learner, a, s):
    if learner.s_dim() == 1:
        log_vals_step = [s]
    elif learner.s_dim() > 1:
        log_vals_step = [s[k] for k in range(learner.s_dim())]
    if learner.a_dim() == 1:
        log_vals_step.append(a)
    elif learner.a_dim() > 1:
        log_vals_step += [a[k] for k in range(learner.a_dim())]
    return log_vals_step

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+






#@profile
def main(g, setting):

    # ===========================
    # * Set up the environment

    if g.CONTINUIFY:  # env gets special wrapping
        assert g.ENV_IS_A_GAZEBO
        env = cont.ConinuifiedGazebo(g.NR_ACTIONS, g.NR_OBSERVATIONS, g.ENV_NAME,monitor_dir=g.MONITOR_DIR(),
                                     red_fact=10)
    else:
        env = create_env(g.ENV_NAME, g.ENV_IS_A_GAZEBO, g.RENDER_ENV, g.MONITOR_DIR(), g.GYM_MONITOR_EN,
                         g.RANDOM_SEED, nr_actions=g.NR_ACTIONS, nr_obs=g.NR_OBSERVATIONS)

    tf.set_random_seed(g.RANDOM_SEED)


    # ===========================
    # * Set up the learner
    if "ddpg" in setting:
        learner = DDPGLearnerBN(observation_space=env.observation_space, action_space=env.action_space,
                              h_shape=g.SIZE_HIDDEN, a_lr=g.ACTOR_LEARNING_RATE, c_lr=g.CRITIC_LEARNING_RATE)
    elif "vanillaPG" in setting:
        learner = VanillaPGLearner(observation_space=env.observation_space, action_space=env.action_space, h_shape=g.SIZE_HIDDEN, gamma=g.GAMMA)
    else:
        raise NotImplementedError


    #learner.sess = tf_debug.LocalCLIDebugWrapperSession(learner.sess)
    #learner.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    saver = tf.train.Saver()
    if g.RESTORE:
        weights_path = g.WEIGHTS_DIR() + "ep_"+str(g.RESTORE_EP)+"_" + str(g.MINIBATCH_SIZE) +".ckpt"
        saver.restore(learner.sess, weights_path)
        print("Model restored from file: %s" % weights_path)

    # ===========================
    # * Set up logging, if needed
    tf_summarizer, logger, log_variable_names_step, \
        log_variable_names_train, log_variable_names_ep = set_up_logging(learner)

    # Initialize replay memory
    replay_buffer = ReplayBuffer(g.BUFFER_SIZE, g.RANDOM_SEED)

    # ===========================
    # * Start training
    # ===========================

    try:
        train_count = 0
        step_count = 0
        ep_rewards = []

        start_ep = g.RESTORE_EP + 1 if g.RESTORE else 0


        for ep in range(start_ep, g.MAX_EPISODES):

            ep_reward = 0
            av_q_max_ep = 0
            av_q_min_ep = 0
            ep_history = []
            s = env.reset()

            print('#################################################')
            print(s)
            break


            for j in range(g.MAX_EP_STEPS):

                a, other_pred_stats = learner.get_action(s, 1 + ep/100.)  # before: just t=ep

                s2, r, terminated, _ = env.step(a)
                if g.CLIP_REWARDS:
                    r = np.clip(r, -1., 1.)

                if g.RENDER_ENV and ep % g.RENDER_EVERY == 0:
                    s2_img = env.render()


                #  Add experience to buffer
                if g.RAW_R:
                    ## Todo: this might throw errors for spaces with more than one dimension -?
                    replay_buffer.add(np.reshape(s, (learner.s_dim(),)), np.reshape(a, (learner.a_dim(),)), r,
                                      terminated, np.reshape(s2, (learner.s_dim(),)))
                else:
                    ep_history.append([s, a, r, int(terminated), s2])   # we'll cast that list to a np.array, so better only have it contain numbers

                # Keep adding experience to the memory until
                # there are at least minibatch size samples
                prefilled_summaries_train = []
                prefilled_summaries_ep = []
                gradients = None
                if replay_buffer.size() > g.MINIBATCH_SIZE and step_count  >= g.LEARN_START:

                    # Sample batch
                    s_batch, a_batch, r_batch, t_batch, s2_batch = \
                        replay_buffer.sample_batch(g.MINIBATCH_SIZE)
                    if not g.USE_OLD_EXP:
                        replay_buffer.clear()

                    #  Train  #Todo: remove the last parameter, "train_actor", from both learner.train()s again.
                              #Todo:  because it's completely obsolete since I added g.TRAIN_START
                    gradients, summaries, losses, loss_summaries, other_training_stats \
                        = learner.train(s_batch, a_batch, r_batch, t_batch, s2_batch, train_actor=True)
                    train_count += 1

                    if g.DEBUG:
                        if np.isnan(np.max([np.max(gr) for gr in gradients])):
                            print("waah")
                        print('| Grad-mean: %.2f' % (np.mean([np.mean(np.abs(gr))for gr in gradients]) ),
                              '| Grad-max: %.2f' % np.max([np.max(np.abs(gr)) for gr in gradients]))
                    # If the learner doesn't overwrite that function, these
                    #   are just zero
                    q_vals = learner.filter_output_qvals(other_training_stats, other_pred_stats)
                    av_q_max_ep += max(q_vals)
                    av_q_min_ep += min(q_vals)

                    prefilled_summaries_train = summaries + loss_summaries

                    # collect gradients
                    logger.collect(dict(zip(learner.gradient_names, np.absolute(gradients))))
                    logger.collect(dict(zip(learner.loss_names, losses)))

                    # collect any other training variables
                    logger.collect(dict(zip(log_variable_names_train, other_training_stats)))

                #  Log state, action etc every step
                log_vals_step = list_state_and_action(learner, a, s)
                log_vals_step += other_pred_stats

                if g.LOG and ep % g.LOG_EVERY == 0:
                    logger.track(dict(zip(log_variable_names_step, log_vals_step)), step_nr=j)

                s = s2
                ep_reward += r
                step_count += 1


                log_vals_ep = []

                if terminated:
                    ep_rewards.append(ep_reward)            #  ["min_q_val", "max_q_val","survival_time", "ep_reward"]
                    log_vals_ep = [av_q_min_ep / float(j), av_q_max_ep / float(j), j, ep_reward, train_count, np.mean(ep_rewards)]

                tf_summarizer.log(learner.sess, ep, j, var_vals_ep=log_vals_ep, var_vals_step=log_vals_step,terminated=terminated,
                                   prefilled_summaries_ep=prefilled_summaries_ep, prefilled_summaries_step=prefilled_summaries_train)

                if terminated:

                        print('| Reward: %.2i' % int(ep_reward), " | Episode", ep,
                              '| Qmax: %.4f' % (av_q_max_ep / float(j)))
                        if not gradients is None:
                            print('| Grad-mean: %.2f' % (np.mean([np.mean(np.abs(gr)) for gr in gradients])),
                                  '| Grad-max: %.2f' % np.max([np.max(np.abs(gr)) for gr in gradients]))

                        # add decayed rewards only once per episode
                        if not g.RAW_R:
                            ep_history = np.array(ep_history)
                            ep_history[:, 2] = discount_rewards(ep_history[:, 2], g.GAMMA)
                            for [s, a, r, term, s2] in ep_history:
                                replay_buffer.add(np.reshape(s, (learner.s_dim(),)), np.reshape(a, (learner.a_dim(),)),
                                                  r, term, np.reshape(s2, (learner.s_dim(),)))

                        # track per-episode variables
                        logger.track(dict(zip(log_variable_names_ep, log_vals_ep)), ep)

                        # store plots for past episode
                        if g.LOG and ep % g.LOG_EVERY == 0 and log_vals_step != []:
                            logger.store_tracked(log_variable_names_step, title="ep-"+str(ep), step_name="steps")
                        # Store in-between versions of 'overall' plots
                        if ep % g.LOG_EVERY == 0 and ep != 0:
                            logger.store_tracked(None, title="until-ep-"+str(ep), step_name="episodes", keep=True)
                            logger.store_tracked_stats(None, title="until-ep-"+str(ep), step_name="episodes", keep=True)

                        # track gradient statistics, if any training was done in this episode
                        #if not gradients is None:
                        if all([key in logger.dict_stats_collecting for key in learner.gradient_names]):
                            logger.merge_collected(learner.gradient_names, ep)
                            logger.merge_collected(learner.loss_names, ep)
                            logger.merge_collected(log_variable_names_train, ep)

                        break

            #
            if g.ENV_IS_A_GAZEBO:
                env._flush(force=True)

            if ep != 0 and g.STORE_WEIGHTS:
                if ep % g.STORE_WEIGHTS_EVERY == 0:
                    if not os.path.exists(g.WEIGHTS_DIR()):
                        os.makedirs(g.WEIGHTS_DIR())
                    saver.save(learner.sess, g.WEIGHTS_DIR() + "ep_"+str(ep)+"_" + str(g.MINIBATCH_SIZE) +".ckpt")



    finally:

        # After training finished, store plots of all per-episode variables & stats
        try:
            logger.store_all("End-ep-"+str(ep), ep, step_name="episode")
            g.store_globals_txt(logger.logdir + "/globalvars.txt")

        finally:

            if g.STORE_WEIGHTS:
                if not os.path.exists(g.WEIGHTS_DIR()):
                    os.makedirs(g.WEIGHTS_DIR())
                try:
                    saver.save(learner.sess, g.WEIGHTS_DIR() + "ep_" + str(ep) + "_" + str(g.MINIBATCH_SIZE) +".ckpt")
                    print "Stored final weights in: "+g.WEIGHTS_DIR() + "ep_" + str(ep) + "_" + str(g.MINIBATCH_SIZE) + ".ckpt"
                except:
                    saver.save(learner.sess, g.WEIGHTS_DIR() + "ep_unknown_" + "_" + str(g.MINIBATCH_SIZE) + ".ckpt")
                    print "Stored final weights in: " + g.WEIGHTS_DIR() + "ep_unknown_" + "_" + str(g.MINIBATCH_SIZE) + ".ckpt"


            # ===========================
            # * Shut down
            # ===========================
            if g.GYM_MONITOR_EN:
                env.monitor.close()

            if g.ENV_IS_A_GAZEBO:
                from subprocess import call
                call(["killall -9 gzserver gzclient roslaunch rosmaster"], shell=True)



if __name__ == '__main__':
    # ===========================
    # * Choose your setting.
    # ===========================

    #setting = "ddpg_pendulum"
    #g = GlobalVars().set_ddpg_pendulum()

    setting = "ddpg_gazebo"
    g = GlobalVars().set_ddpg_gazebo()

    # setting = "vanillaPG_cartpole"
    # from globals_vanillaPG_cartpole import *

    # setting = "vanillaPG_gazebo"
    # g = GlobalVars().set_vanillaPG_gazebo()
    # from globals_vanillaPG_gazebo import *

    #  todo
    # setting = "ddpg_pendulum_from_pixels"
    # setting = "DQN_cartpole"
    # setting = "DQN_gazebo"



    # g: container for global variables
    main(g, setting)