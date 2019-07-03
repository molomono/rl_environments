#TODO: This environment provides rewards the AI for balancing the robot

from .balancebot_vrep_env_noise import BalanceBotVrepEnvNoise
from scripts.util import *
import numpy as np

class BalanceBotVrepEnvBalance(BalanceBotVrepEnvNoise):
	def compute_reward(self):
		''' This function calculates the reward based on the observation list.

		Reward := (w_1 * a + w_2 * b + w_3 * c + ....) / (sum(w_1, w_2, w_3, ....)
		In this function a, b, c are different values calculated from the observation list these 
		values are the quantificaiton of desired behavior. The values have a maximum of 1.
		w_x is the weight for each value and is used to indicate the priority of the different values

		:returns: reward value
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
		r_regul = np.sum(gaussian(delta_pos, sigma=2.0)) * 1/float(len(delta_pos))
		
		#calculate the reward function
		r_alive = 1.0
		w = [10., 4., 0.]
		scale_factor = 1./sum(w)
		return (w[0] * r_alive + w[1] * (1. - norm_pos_dist) + w[2] * r_regul )* scale_factor

	def compute_action(self, action):
		''' Transform the action vector 
.
		This version of thefunction translate the predicted action into translation.
		
		:param action: predicted action
		:returns: Action in the form of ratio of motor power -1 to 1
		'''
		kinematics = np.matrix([[0., 0.], [1., 1.]]) 
		return np.asarray(np.matrix(action) * kinematics).reshape(-1)

	def sample_goal(self):
		''' Samples the goal space for an XY coordinate but returns only the Y as a goal

		The goal generated is only for learning to balance and translate, not including rotation.
		
		:returns: numpy float array 
		'''
		goal = self.goal_space.sample()
		return np.hstack([0.0, goal[1]])