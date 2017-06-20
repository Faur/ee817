
import numpy as np
import tensorflow as tf
import tflearn
import gym
from learner import *




# ===========================
#   Actor and Critic DNNs
# ===========================

# !! Not yet the DDPGLearner class !! That one is further down. (note-to-self)
class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, h_shape=[20,20]):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.h_shape = h_shape

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)

        self.summary_grads = []
        for grad in self.actor_gradients:
            self.summary_grads.append(tf.summary.histogram("Actor_Gradients_" + grad.name, grad))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

        #self.ep_ave_max_q = 0


    def create_actor_network(self):
        inputs = tf.placeholder(tf.float32, shape=[None, self.s_dim])
        net = tflearn.batch_normalization(inputs)
        for i in range(0, len(self.h_shape)):
            net = tflearn.fully_connected(net, self.h_shape[i], activation='relu')
            net = tflearn.batch_normalization(net, gamma=1)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # tflearn.fully_connected()
        # tf.contrib.layers
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        return self.sess.run([self.optimize, self.actor_gradients, self.summary_grads], feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    # just get the gradients, don't train on them
    def dont_train(self, inputs, a_gradient):
        return self.sess.run([self.actor_gradients, self.summary_grads], feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars, h_shape=[20,20]):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.h_shape = h_shape

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i],
                                                                            1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.gradients = self.optimizer.compute_gradients(self.loss)
        self.gradients_only = tf.gradients(self.loss, self.network_params)
        self.gradients_only =  [x for x in self.gradients_only if x is not None]
        self.optimize = self.optimizer.apply_gradients(self.gradients)

        self.summary_grads = []
        for grad in self.gradients:
            if grad[0] != None:
                self.summary_grads.append(tf.summary.histogram("Critic_Gradients_" + grad[1].name, grad[0]))
        self.summary_loss = tf.summary.scalar("Critic_loss", self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self, h_shape=[20, 20]):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.batch_normalization(inputs)
        for i in range(0, len(self.h_shape)-1):
            net = tflearn.fully_connected(net, self.h_shape[i], activation='relu')
            net = tflearn.batch_normalization(net)
        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, self.h_shape[-1])
        t2 = tflearn.fully_connected(action, self.h_shape[-1])

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        grad_vars = [grad[1] for grad in self.gradients]
        return self.sess.run([self.out, self.optimize, self.gradients_only, self.summary_grads, self.loss, self.summary_loss], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


# ===========================
#   Tensorflow Summary Ops
# ===========================


def build_summaries(state_dim):
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
    for l in range(state_dim):
        state_var = tf.Variable(0.)
        summary_state_var = tf.summary.scalar("State_" + str(l), state_var)
        summary_vars_step.append(state_var)
        summary_ops_step.append(summary_state_var)
    q_summary = tf.summary.scalar("Q", q_value)
    summary_ops_step.append(q_summary)
    summary_ops_step = tf.summary.merge(summary_ops_step)

    return summary_ops_ep, summary_vars_ep, summary_ops_step, summary_vars_step  ## ep: log them per episode, step: per step











# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



class DDPGLearnerBN(Learner):
    '''
    Encapsulates all nets and function belonging to the DDPG actor-critic method.
    Owns a tensorflow session.
    '''
    def __init__(self, observation_space, action_space, a_lr=0.0001, c_lr=0.001, gamma=0.99,
                 tau=0.001, h_shape=[20, 20]):

        self.action_space = action_space
        self.observation_space = observation_space

        self.actor_learning_rate = a_lr
        self.critic_learning_rate = c_lr
        # Discount factor
        self.gamma = gamma
        # Soft target update param
        self.tau = tau

        self.sess = tf.Session()

        state_dim = self.s_dim()
        action_dim = self.a_dim()
        action_bound = self.action_space.high
        # Ensure action bound is symmetric or starts at 0 - for scaling
        assert (self.action_space.high == -self.action_space.low) or (self.action_space.low == 0)

        self.actor = ActorNetwork(self.sess, state_dim, action_dim, action_bound,
                             self.actor_learning_rate, self.tau, h_shape=h_shape)

        self.critic = CriticNetwork(self.sess, state_dim, action_dim, self.critic_learning_rate,
                              self.tau, self.actor.get_num_trainable_vars(), h_shape=h_shape)

        # Set up summary Ops
        summary_ops_ep, summary_vars_ep, summary_ops_step, summary_vars_step = build_summaries(state_dim)

        self.sess.run(tf.global_variables_initializer())

        # Initialize target network weights
        tau_tmp = self.tau
        self.tau = 1.
        self.actor.update_target_network()
        self.critic.update_target_network()
        self.tau = tau_tmp
        del tau_tmp

        self.loss_names = ["critic_loss"]
        self.other_training_stats_names = ["predicted_q_values"]
        self.other_prediction_stats_names = []

        self.gradient_names = []
        for grad in self.actor.actor_gradients:
            self.gradient_names.append("Actor_Grad_"+grad.name)
        for grad in self.critic.gradients_only:
            self.gradient_names.append("Critic_Grad_"+grad.name)



    # For using "with Learner(..) as learner: "
    # See https://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.close()

    def __del__(self):
        self.sess.close()


    def get_graph(self):
        return self.sess.graph


    def train(self, s_batch, a_batch, r_batch, t_batch, s2_batch, train_actor=True):

        minibatch_size = s_batch.shape[0]

        # Calculate targets
        target_q = self.critic.predict_target(
            s2_batch, self.actor.predict_target(s2_batch))

        y_i = []
        for k in range(minibatch_size):
            if t_batch[k]:
                y_i.append(r_batch[k])
            else:
                y_i.append(r_batch[k] + self.gamma * target_q[k])

        # Update the critic given the targets
        predicted_q_value, _, crit_grads, summary_crit_grad, critic_loss, summary_crit_loss = self.critic.train(
            s_batch, a_batch, np.reshape(y_i, (minibatch_size, 1)))

        #self.ep_ave_max_q += np.amax(predicted_q_value)

        # Update the actor policy using the sampled gradient
        a_outs = self.actor.predict(s_batch)
        a_grads = self.critic.action_gradients(s_batch, a_outs)
        if train_actor:
            _, actor_grads, summary_actor_grad = self.actor.train(s_batch, a_grads[0])
        else:
            actor_grads, summary_actor_grad = self.actor.dont_train(s_batch, a_grads[0])

        # Update target networks
        self.actor.update_target_network()
        self.critic.update_target_network()

        all_grads = actor_grads + crit_grads
        all_grad_summaries = summary_actor_grad + summary_crit_grad
        assert len(all_grads) == len(self.gradient_names) == len(all_grad_summaries)


        return all_grads, all_grad_summaries, [critic_loss], [summary_crit_loss], [predicted_q_value]







    def get_action(self, s, t=0):
        '''
        :param s: state / observation
        :param t: optional, timestep, for decreasing exploration noise over time
        :return: the advised action & empty "other prediction stats"
        '''
        # Added exploration noise
        a = self.actor.predict(np.reshape(s, (1, self.actor.s_dim))) # + np.random.uniform(self.action_space.low * 1. / (1. + t),
                                                                     #          self.action_space.high * 1. / (1. + t))

        #if t % 20 == 0:
        #    a = self.actor.predict(np.reshape(s, (1, self.actor.s_dim)))
        #else:
        #    a = np.random.uniform(self.action_space.low, self.action_space.high)

        a = np.clip(a, self.action_space.low, self.action_space.high)

        return a.flatten(), []          # flatten: actor.predict returns an array of shape (1,1), but logging expects a single-dimensional input



    def filter_output_qvals(self, other_training_stats, other_prediction_stats):
        return other_training_stats[0]