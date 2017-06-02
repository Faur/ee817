
import numpy as np
import tensorflow as tf

##       - a custom log function, to modify/overwrite in a new instance, that logs only a few of the summaries,
#           depending on the current indices.
#       - two types of summaries: those filled with a value manually, and those where we directly get the summary sting to log
#           (no class needed for that...)
#       - maybe this class would be only an outsourcing of a few lines of code, good for readability.

class SummaryManager():

    def __init__(self, logdir, sess, graph=None, variable_names_ep=[], variable_names_step=[]):   #, prefilled_summaries=[]):

        self.writer = tf.summary.FileWriter(logdir, graph)
        self.variables_ep = []
        self.variable_names_ep = variable_names_ep
        summaries_ep = []
        self.variables_step = []
        self.variable_names_step = variable_names_step
        summaries_step = []
        # self.prefilled_summaries = prefilled_summaries

        for var_n in variable_names_ep:
            var = tf.Variable(0.)
            self.variables_ep.append(var)
            summaries_ep.append(tf.summary.scalar(var_n, var))
        for var_n in variable_names_step:
            var = tf.Variable(0.)
            self.variables_step.append(var)
            summaries_step.append(tf.summary.scalar(var_n, var))

        self.summary_ops_ep = None if len(summaries_ep) == 0 else tf.summary.merge(summaries_ep)
        self.summary_ops_step = None if len(summaries_step) == 0 else tf.summary.merge(summaries_step)

        self.log_counter = 0

        sess.run(tf.global_variables_initializer())


    def log(self, sess, episode, step=None, var_vals_ep=[], var_vals_step=[], terminated=False, prefilled_summaries_ep=[],
            prefilled_summaries_step=[], printout=False):
        '''
        Log the variables here.
        :param variable_list: pass the variable values to log here
        :param terminated: whether the episode terminates after this step
        :param var_vals_ep, prefilled_summaries_ep: are logged whenever terminated==True
        :return: nothing
        '''
        if terminated and len(prefilled_summaries_ep) > 0:
            for sum in prefilled_summaries_ep:
                self.writer.add_summary(sum, episode)

        if len(prefilled_summaries_step) > 0:
            assert step != None
            for sum in prefilled_summaries_step:
                self.writer.add_summary(sum, step)

        if terminated and len(var_vals_ep) > 0:
            assert len(var_vals_ep)==len(self.variable_names_ep)
            var_vals_ep = [np.asscalar(var) if type(var) == np.ndarray else var for var in var_vals_ep]
            dict_to_feed = {}
            for k, var in enumerate(var_vals_ep):
                dict_to_feed[self.variables_ep[k]] = var
            summary_str = sess.run(self.summary_ops_ep, feed_dict=dict_to_feed)
            self.writer.add_summary(summary_str, episode)

        if not len(var_vals_step)==0:
            assert len(var_vals_step)==len(self.variable_names_step)
            var_vals_step = [np.asscalar(var) if type(var) == np.ndarray else var for var in var_vals_ep]
            dict_to_feed = {}
            for k, var in enumerate(var_vals_step):
                dict_to_feed[self.variables_step[k]] = var     # if error here, the variable passed was not a scalar.
            summary_str = sess.run(self.summary_ops_step, feed_dict=dict_to_feed)
            self.writer.add_summary(summary_str, self.log_counter)

        if printout:
            pass ## todo: print out what was logged

        self.writer.flush()
        self.log_counter += 1




    # X: summary or variable-name
    def add_summary(self, X, prefilled=False, ep=True):
        if prefilled:
            self.prefilled_summaries.append(X)
        else:
            var = tf.Variable(0.)
            if ep:
                self.variable_names_ep.append(X)
                self.variables_ep.append(var)
                self.summaries_ep.append(tf.summary.scalar(var))
                self.summary_ops_ep = tf.summary.merge(self.summaries_ep)
            else:
                self.variable_names_step.append(X)
                self.variables_step.append(var)
                self.summaries_step.append(tf.summary.scalar(var))
                self.summary_ops_step = tf.summary.merge(self.summaries_step)
        return

    # only for non-prefilled summaries
    # def add_op_set(self, name_list, set_name):
    #     sums = []
    #     for n in name_list:
    #         index = self.variable_names.index(n)
    #         assert index >= 0
    #         sums.append(self.summaries[index])
    #     op_set = tf.summary.merge(sums)
    #     self.op_sets[set_name : op_set]
    #     return op_set





class SummaryManagerDDPG(SummaryManager):


    def __init__(self, logdir, env, graph=None):
        super(SummaryManagerDDPG, self).__init__(logdir, graph=graph)

        episode_reward = tf.Variable(0.)
        s_rew = tf.summary.scalar("Reward", episode_reward)
        episode_ave_max_q = tf.Variable(0.)
        s_qmax = tf.summary.scalar("Qmax Value", episode_ave_max_q)
        self.summary_ops_ep = tf.summary.merge([s_rew, s_qmax])
        self.variables_ep = [episode_reward, episode_ave_max_q]
        self.variable_names_ep = [s_rew.name, s_qmax.name]

        chosen_action = tf.Variable(0.)
        a_summary = tf.summary.scalar("Action", chosen_action)
        q_value = tf.Variable(0.)
        self.summary_ops_step = [a_summary]
        self.variables_step = [chosen_action]
        self.variable_names_step = [a_summary.name]
        for l in range(env.observation_space.shape[0]):
            state_var = tf.Variable(0.)
            summary_state_var = tf.summary.scalar("State_" + str(l), state_var)
            self.variables_step.append(state_var)
            self.summary_ops_step.append(summary_state_var)
            self.variable_names_step.append(summary_state_var.name)
        q_summary = tf.summary.scalar("Q", q_value)
        self.variables_step.append(q_value)
        self.summary_ops_step.append(q_summary)
        self.variable_names_step.append(q_summary.name)
        self.summary_ops_step = tf.summary.merge(self.summary_ops_step)



    # variable_dict: name-value pair. names are keys.
    def log(self, sess, episode, step=None, var_vals_ep=[], var_vals_step=[], terminated=False, prefilled_summaries=[], printout=False):

        if step==20:
            for sum in prefilled_summaries:
                self.writer.add_summary(sum, episode)

        if len(prefilled_summaries)==0:
            assert step!=None

        if terminated and not len(var_vals_ep)==0:
            assert len(var_vals_ep)==len(self.variable_names_ep)
            dict_to_feed = {}
            for k, var in enumerate(var_vals_ep):
                dict_to_feed[self.variables_ep[k]] = var
            summary_str = sess.run(self.summary_ops_ep, feed_dict=dict_to_feed)
            self.writer.add_summary(summary_str, episode)

        if not len(var_vals_step)==0:
            assert len(var_vals_step)==len(self.variable_names_step)
            dict_to_feed = {}
            for k, var in enumerate(var_vals_step):
                dict_to_feed[self.variables_step[k]] = var
            summary_str = sess.run(self.summary_ops_step, feed_dict=dict_to_feed)
            self.writer.add_summary(summary_str, self.log_counter)

        if printout:
            pass ## todo: print out what was logged

        self.writer.flush()
        self.log_counter += 1


    def build_summaries(self, env):
        episode_reward = tf.Variable(0.)
        s_rew = tf.summary.scalar("Reward", episode_reward)
        episode_ave_max_q = tf.Variable(0.)
        s_qmax = tf.summary.scalar("Qmax Value", episode_ave_max_q)
        summary_ops_ep = tf.summary.merge([s_rew, s_qmax])
        summary_vars_ep = [episode_reward, episode_ave_max_q]

        chosen_action = tf.Variable(0.)
        a_summary = tf.summary.scalar("Action", chosen_action)
        q_value = tf.Variable(0.)
        summary_ops_step = [a_summary]
        summary_vars_step = [chosen_action]
        for l in range(env.observation_space.shape[0]):
            state_var = tf.Variable(0.)
            summary_state_var = tf.summary.scalar("State_" + str(l), state_var)
            summary_vars_step.append(state_var)
            summary_ops_step.append(summary_state_var)
        q_summary = tf.summary.scalar("Q", q_value)
        summary_ops_step.append(q_summary)
        summary_ops_step = tf.summary.merge(summary_ops_step)

        return summary_ops_ep, summary_vars_ep, summary_ops_step, summary_vars_step  ## ep: log them per episode, step: per step



