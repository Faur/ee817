from __future__ import absolute_import, division, print_function, unicode_literals

import threading
# import multiprocessing
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
# import scipy.signal
# %matplotlib inline
# from helper import *
# from vizdoom import *

# from random import choice
from time import sleep
# from time import time

from juliani_helper import *

class AC_Network():
	def __init__(self, s_size, a_size, scope, trainer):
		with tf.variable_scope(scope):
			# self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
			# # Convert input to shape=[batch_size, 84, 84, 1] images
			# self.imageIn = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])

			self.inputs = tf.placeholder(dtype=tf.float32, shape=[None] + s_size, name='inputs')

			## Build model
			if 0: # CNN
				self.conv1 = slim.conv2d(
					inputs=self.inputs,
					activation_fn=tf.nn.elu,
					num_outputs=16,
					kernel_size=[8,8],
					stride=[4,4],
					padding='VALID'
					)
				self.conv2 = slim.conv2d(
					inputs=self.conv1,
					activation_fn=tf.nn.elu,
					num_outputs=32,
					kernel_size=[4,4],
					stride=[2,2],
					padding='VALID'
					)
				hidden = slim.fully_connected(slim.flatten(self.conv2), 256, activation_fn=tf.nn.elu)
			else: # Dense
				hidden = slim.fully_connected(self.inputs, 20, activation_fn=tf.nn.elu)

			# TODO: LSTM part is temporarily disabled!
			with tf.variable_scope('LSTM_cell'):
				lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
				c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
				h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
				self.state_init = (c_init, h_init)
				c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c], name='c_in')
				h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h], name='h_in')
				self.state_in = (c_in, h_in)
				rnn_in = tf.expand_dims(hidden, axis=[0]) # shape = [1, hidden.shape]
				step_size = tf.shape(self.inputs)[:1]
				state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
				lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
					lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
					time_major=False)
				lstm_c, lstm_h = lstm_state
				self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
				rnn_out = tf.reshape(lstm_outputs, [-1, 256])

			#Output layers for policy and value estimation
			# self.policy = slim.fully_connected(rnn_out, a_size,
			# 	activation_fn=tf.nn.softmax,
			# 	weights_initializer=normalized_columns_initializer(0.01),
			# 	biases_initializer=None)
			# self.value = slim.fully_connected(rnn_out, 1, activation_fn=None,
			# 	weights_initializer=normalized_columns_initializer(1.0),
			# 	biases_initializer=None)
			self.policy = slim.fully_connected(hidden, a_size,
				activation_fn=tf.nn.softmax,
				weights_initializer=normalized_columns_initializer(0.01),
				biases_initializer=None)
			self.value = slim.fully_connected(hidden, 1, activation_fn=None,
				weights_initializer=normalized_columns_initializer(1.0),
				biases_initializer=None)

			### Only the worker networks need ops for loss functions and gradient updating
			if scope != 'global':
				self.actions = tf.placeholder(dtype=tf.int32, shape=None, name='actions') # discrete actions!
				self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32, name='actions_hot')
				self.target_v = tf.placeholder(shape=[None], dtype=tf.float32, name='target_v')
				self.advantages = tf.placeholder(shape=[None], dtype=tf.float32, name='advantages')

				self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, axis=[1])

				## Loss function
				self.entropy = - tf.reduce_mean(self.policy*tf.log(self.policy))
				self.policy_loss = - tf.reduce_mean(tf.log(self.responsible_outputs)*self.advantages)
				self.value_loss = 0.5 * tf.reduce_mean(tf.square(self.target_v-tf.reshape(self.value, [-1])))
				# TODO: This loss mixing is extremely env dependent
				self.value_loss *= 0.5
				# self.loss = self.value_loss + self.policy_loss - 0.01*self.entropy
				self.loss = self.value_loss + self.policy_loss 

				## Get gradients from local network using local losses
				local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
				self.gradients = tf.gradients(self.loss, local_vars)
				self.vars_norms = tf.global_norm(local_vars)
				grads, self.grads_norms = tf.clip_by_global_norm(self.gradients, 40.0)

				# Apply gradients to global network
				global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
				self.apply_gradients = trainer.apply_gradients(zip(grads, global_vars))

class Worker():
	def __init__(self, env, number, s_size, a_size, trainer, model_path, global_episodes):
		self.env = env
		self.name = 'worker_' + str(number)
		self.number = number
		self.model_path = model_path
		self.trainer = trainer
		self.global_episodes = global_episodes
		self.increment = self.global_episodes.assign_add(1) #global_episodes is a tf.Variable
		self.episode_rewards = []
		self.episode_lengths = []
		self.episode_mean_values = [] # Q: what is this?
		self.summary_writer = tf.summary.FileWriter('train_' + str(self.number))

		# Create local copy of network
		self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
		# Copy global prameters to local network
		self.update_local_ops = update_target_graph('global', self.name)

		self.actions = np.identity(a_size, dtype=bool).tolist() # TODO: This is in a weird format!

	def train(self, rollout, sess, gamma, bootstrap_value):
		""" Given a minibatch of experiences update the global parameters"""

		rollout = np.array(rollout) # episode_buffer: list([s, a, r, s1, done, v[0,0]])
									#                       0  1  2   3                 4       5
		observations = rollout[:, 0] # Q: What is in this? how does this work?
		observations = np.array([i for i in observations])
		actions = rollout[:, 1]
		rewards = rollout[:, 2]
		next_observations = rollout[:, 3]
		next_observations = np.array([i for i in next_observations])
		values = rollout[:, 5]

		## Compute advantage and discounted returns
		## We use the "Generalized Advantage Estimation"
		# TODO: Look up GAE
		self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value]) 
			# TODO: Verify: Containsr_t and v_(t+1)
			# Q: What does this look like?
		discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
		self.values_plus = np.asarray(values.tolist() + [bootstrap_value])
		advantages = rewards + gamma*self.values_plus[1:] - self.values_plus[:-1]
		advantages = discount(advantages, gamma)

		## Update the global network using gradients from loss
		## Generate netowrk statistics
		# print('observations')
		# print(type(observations))
		# print(observations.shape)
		# print(observations[0].shape)
		# print('values')
		# print(type(values))
		# print(values.shape)

		rnn_state = self.local_AC.state_init
		feed_dict = {
			self.local_AC.target_v : discounted_rewards,
			# self.local_AC.inputs : np.vstack(observations),
			self.local_AC.inputs : observations,
			self.local_AC.actions : actions,
			self.local_AC.advantages : advantages,
			self.local_AC.state_in[0] : rnn_state[0],
			self.local_AC.state_in[1] : rnn_state[1],
		}

		v_l, p_l, e_l, g_n, v_n, _ \
			= sess.run([
					self.local_AC.value_loss,
					self.local_AC.policy_loss,
					self.local_AC.entropy,
					self.local_AC.grads_norms,
					self.local_AC.vars_norms,
					self.local_AC.apply_gradients
				], 
				feed_dict = feed_dict
		)
		# TODO: I use mean, not sum. Should we still divide with len(rollout?)
		return v_l/len(rollout), p_l/len(rollout), e_l/len(rollout), g_n, v_n

	def env_stepper(self, a, num_repeat=4):
		s, r, d, i = self.env.step(a)
		r_sum = r
		for i in range(num_repeat-1):
			if d: break
			s, r, d, i = self.env.step(a)
			r = np.clip(r, -1, 1)
			r_sum += r

		return s, r_sum, d, i


	def work(self, max_episode_length, gamma, sess, coord, saver):
		episode_count = sess.run(self.global_episodes)
		total_steps = 0
		print("Staring worker_" + str(self.number))
		with sess.as_default(), sess.graph.as_default():
			while not coord.should_stop():
				print(str(episode_count) + ': worker_' + str(self.number))
				sess.run(self.update_local_ops) # set local param to value of global
				episode_buffer = [] # TODO: Create a class instead of using lists
				episode_values = []
				episode_frames = []
				episode_reward = 0
				episode_step_count = 0
				done = False

				s = self.env.reset() # Initial observation
				# self.env.render(mode='rgb_array')
				episode_frames.append(s)
				# s = prepro(s, down_sample_factor)

				rnn_state = self.local_AC.state_init # Tuple of arrays of zero

				while not done:

					a_dist, v, rnn_state = \
						sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
							feed_dict={
								self.local_AC.inputs : [s],
								self.local_AC.state_in[0] : rnn_state[0], # c
								self.local_AC.state_in[1] : rnn_state[1]  # h
							}
						)
					# TODO: Is this really a sensical way of picking a?
					a = np.random.choice(a_dist[0], p=a_dist[0]) # Q: what does a_dist look like?
					a = np.argmax(a_dist == a)

					# r = self.env.make_action(self.actions[a]) / 100.0 # TODO: Fix this hardcoding!
					s1, r, done, _ = self.env_stepper(a, num_repeat=1)
					if 0 and self.name == 'worker_0':
						self.env.render()

					if not done:
						# s1 = self.env.get_state().screen_buffer
						# s1 = self.env.render(mode='rgb_array')
						episode_frames.append(s1)
						# s1 = process_frames(s1)
						# s1 = process_frames(s1)
					else:
						s1 = s

					episode_buffer.append([s, a, r, s1, done, v[0,0]]) # Q: What is this used for?
						# Q: Do we really need both s and s1? 
					episode_values.append(v[0,0])
					episode_reward += r
					total_steps += 1
					episode_step_count += 1
					s = s1

					# If experience buffer is full: perform update to global parameters
					if len(episode_buffer) == 30 \
							and not done \
							and episode_step_count != max_episode_length - 1:
						v1 = sess.run(self.local_AC.value,
							feed_dict={
								self.local_AC.inputs : [s],
								self.local_AC.state_in[0] : rnn_state[0],
								self.local_AC.state_in[1] : rnn_state[1]
							}
						)

						# Update global parameters
						v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
						episode_buffer = []
						# Update local parameters
						sess.run(self.update_local_ops)

				## [Episode finished]
				self.episode_rewards.append(episode_reward)
				self.episode_lengths.append(episode_step_count)
				self.episode_mean_values.append(np.mean(episode_values))


				## Update global parameters
				if len(episode_buffer) != 0:
					v1 = sess.run(self.local_AC.value,
						feed_dict={
							self.local_AC.inputs : [s],
							self.local_AC.state_in[0] : rnn_state[0],
							self.local_AC.state_in[1] : rnn_state[1]
						}
					)
					v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)


				## Monitor progress
				monitor_frequency = 5
				if episode_count % monitor_frequency == 0 and episode_count > 0:
					if self.name == 'worker_0':
						# if episode_count % (monitor_frequency*5) == 0:
						# 	print(str(episode_count) + ': Gif created.')
						# 	time_per_step = 0.05
						# 	images = np.array(episode_frames)
						# 	make_gif(images, './frames/image' + str(episode_count) + '.gif',
						# 		duration=len(images)*time_per_step, true_image=True, salience=False)
						if episode_count % (monitor_frequency*50) == 0:
							saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
							print("Model saved.")

					## Gather information about the last monitor_frequency episodes
					mean_reward = np.mean(self.episode_rewards[-5:])
					mean_length = np.mean(self.episode_lengths[-5:])
					mean_value = np.mean(self.episode_mean_values[-5:])

					## Write summaries
					summary = tf.Summary()
					summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
					summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
					summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
					summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
					summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
					summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
					summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
					summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
					self.summary_writer.add_summary(summary, episode_count)
					# TODO: how is summary from the different workers handled? 
					# 		Do we really need to track the output of all of them?

					self.summary_writer.flush()
				if self.name == 'worker_0': # Add one to the global episode counter
					sess.run(self.increment)
				episode_count += 1





if __name__ == '__main__':
	print('\na3c.py terminated successfully')