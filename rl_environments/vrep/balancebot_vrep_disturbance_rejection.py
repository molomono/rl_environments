#Incomplete environment

#TODO: This environment is an extension to the balance task in which the robot is
# subjected to uniformly sampled forces at random timesteps.
# This is to ensure the robot is subjected to disturbances during training.

#TODO: This environment provides rewards the AI for balancing the robot


from .balancebot_vrep_env_noise import BalanceBotVrepEnvNoise
from scripts.util import *
import numpy as np

class BalanceBotVrepEnvDisturbance(BalanceBotVrepEnvNoise):

	def compute_reward(self):
		'''Calculates the dense reward for the environment and adds a sparse reward for achieving the goal
		
		If the goal has been achieved a new goal is sampled.

		:returns: reward (sparse) + reward (goal_achievement)
		'''
		
		# Calculate the goal vector relative to the position of the balance-bot
		rel_pos_dist = np.linalg.norm([self.goal[0]-self.observation[9], self.goal[1]-self.observation[10]])
		# Normalize the goal vector with respect to the largest paussible goal distance
		norm_pos_dist = np.asarray(rel_pos_dist * 1./np.linalg.norm([self.goal_max,self.goal_max]) ).reshape(-1)[0]
		# Print the goal distance if verbosity is on
		if self.verbose:
			print("Relative X: {} Y: {} Dist: {}".format(self.observation[-3], self.observation[-2], self.observation[-1]))
		
		# Regulatory factor for wheel velocities
		delta_pos = np.asarray([self.l_wheel_delta, self.r_wheel_delta])
		
		# The actual reward function is below, a standard format is being used to ensure the size of the reward remains predictable:
		# y_reward := (w_1 * a + w_2 * b + w_3 * c + ....) / (sum(w_1, w_2, w_3 ....))
		# a, b, c are different attributes that provide a reward, they do this on a scale no larger than 1
		# w_x is the weight for each attribute this provides the priority to different learned attributes
		# The sum of weights at the end is used to ensure that the max reward that can be recieved is 1.0
		r_alive = 1.0
		w = [5., 1.]
		scale_factor = 1./sum(w)
		return (w[0] * r_alive + w[1] * (1. - norm_pos_dist) )* scale_factor

	def compute_action(self, action):
		''' Transform the action vector 
		This version of thefunction translate the predicted action into translation and rotation.
		
		:param action: predicted action
		:returns: Action in the form of ratio of motor power -1 to 1
		'''
		kinematics = np.matrix([[0., 1.], [1., 1.]]) 
		return np.asarray(np.matrix(action) * kinematics).reshape(-1)

	def sample_goal(self):
		''' Samples the goal space for an XY coordinate but returns only the Y as a goal

		The goal generated is only for learning to balance and translate, not including rotation.
		
		:returns: numpy float array 
		'''
		return np.hstack([0.0, 0.0])

	def step(self, action):
		self.disturbance()
		return super().step(action)

	def disturbance():
		magnitude = [10, 5, 1]
		
