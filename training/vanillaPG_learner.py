import numpy as np
import tensorflow as tf
import datetime
import tensorflow.contrib.slim as slim
from learner import *

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.layers.advanced_activations import LeakyReLU


try:
    xrange = xrange
except:
    xrange = range


def setup():
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = '/tmp/gazebo_gym_experiments/dqn_'+timestr
    weights_path = '/tmp/turtle_c2_vPG.h5'
    monitor_path = '/tmp/turtle_c2_vPG'
    params_json  = '/tmp/turtle_c2_vPG.json'


class agent():
    def __init__(self, lr, s_size, a_size, hiddenLayers):
        # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        self.input_size = s_size
        self.output_size = a_size

        # hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        ## MAKE MODEL BIGGER!
        hidden = self.createModel(self.state_in, hiddenLayers, "LeakyReLU")

        self.output = slim.fully_connected(hidden(self.state_in), a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        # The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)
        #self.gradients_only = [x for x in self.gradients if x is not None]

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        #self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))
        self.update_batch = optimizer.apply_gradients(zip(self.gradients, tvars))

        self.summary_loss = tf.summary.scalar("Actor_Loss", self.loss)
        self.summary_grads = []
        #for grad in self.gradients_only:
        #    self.summary_grads.append(tf.summary.histogram("Actor_Gradients_" + grad.name, grad))

    def createModel(self, inputs, hiddenLayers, activationType):
        model = Sequential()
        if len(hiddenLayers) == 0: 
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform'))
            model.add(Activation("linear"))
        else :
#             model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform'))
            model.add(Dense(hiddenLayers[0], kernel_initializer="lecun_uniform", input_shape=(self.input_size,))) #ccsmm            
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))
            
            for index in range(1, len(hiddenLayers)):
                # print("adding layer "+str(index))
                layerSize = hiddenLayers[index]
#                 model.add(Dense(layerSize, init='lecun_uniform'))
                model.add(Dense(layerSize, kernel_initializer='lecun_uniform')) #ccsmm
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
#             model.add(Dense(self.output_size, init='lecun_uniform'))
            model.add(Dense(self.output_size, kernel_initializer='lecun_uniform')) #ccsmm            
            model.add(Activation("linear"))
        # optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        # model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



class VanillaPGLearner(Learner):
    def __init__(self, observation_space, action_space, hiddenLayers, gamma=0.99):

        tf.reset_default_graph()  # Clear the Tensorflow graph.

        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        #self.update_frequency = update_freq  ## vary batchsize instead

        self.myAgent = agent(lr=0.01, s_size=self.s_dim(), a_size=self.get_action_count(), hiddenLayers=hiddenLayers)  # Load the agent.

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        self.gradBuffer = self.sess.run(tf.trainable_variables())
        for ix, grad in enumerate(self.gradBuffer):
            self.gradBuffer[ix] = grad * 0

        self.loss_names = ["Actor_Loss"]
        self.other_training_stats_names = []
        self.other_prediction_stats_names = ["Action_Probabs"]

        self.gradient_names = []
        for grad in self.myAgent.gradients:
            self.gradient_names.append("Actor_Grad_" + grad.name)




    def train(self, s_batch, a_batch, r_batch, t_batch, s2_batch):
        ''' Update the learner's parameters according to the training input.
        :param s_batch: a batch of states, should fit self.input_space
        :param a_batch: acctions taken, at each state in s_batch
        :param r_batch: corresponding rewards - depending on the learner, pure rewards, or accumulated rewards
        :param t_batch: whether the environment terminated after this step
        :param s2_batch: state we ended up in
        :return: gradients: all gradients' values during this update; list of numpy-matrices
        :return: summaries: list with any summary strings that were created during the update
        :return: other_training_stats:  See comment in "learner.py", on top
        '''

        feed_dict = {self.myAgent.reward_holder: r_batch, self.myAgent.action_holder: np.squeeze(a_batch),
                     self. myAgent.state_in: s_batch}
        #grads, gradients, grad_summaries, loss, loss_summary = \
        #    self.sess.run([self.myAgent.gradients, self.myAgent.gradients_only,
        #                 self.myAgent.summary_grads, self.myAgent.loss, self.myAgent.summary_loss],
        #                  feed_dict=feed_dict)
        #for idx, grad in enumerate(grads):
        #    self.gradBuffer[idx] += grad

        #feed_dict = dict(zip(self.myAgent.gradient_holders, self.gradBuffer))
        #_ = self.sess.run([self.myAgent.update_batch], feed_dict=feed_dict)
        _, grads, grad_summaries, loss, loss_summary  \
            = self.sess.run([self.myAgent.update_batch, self.myAgent.gradients,
                         self.myAgent.summary_grads, self.myAgent.loss, self.myAgent.summary_loss],
                            feed_dict=feed_dict)
        #for ix, grad in enumerate(self.gradBuffer):
        #        self.gradBuffer[ix] = grad * 0


        #return gradients, summaries, losses, loss_summaries, other_training_stats
        return grads, grad_summaries, [loss], [loss_summary], []



    def get_action(self, s, t=1):
        '''
        Output an advised action.
        :param s: the state, should fit self.input_space
        :param t: optional timestep, for modulating exploration noise or similar
        :return: action, should fit self.action_space
        :return: other_prediction_stats: an empty list, or a list with other net statistics observed in this action estimation
        '''
        # Probabilistically pick an action given our network outputs.
        a_dist = self.sess.run(self.myAgent.output, feed_dict={self.myAgent.state_in: [s]})
        a = np.random.choice(range(self.get_action_count()), p=a_dist[0])
        if a_dist[0,a]==0:
            print("waahhaaaha")
        if not np.isscalar(a):
            a = a.flatten()

        return a,  [a_dist[0].flatten()]
