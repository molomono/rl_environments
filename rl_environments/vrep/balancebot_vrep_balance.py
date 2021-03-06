from .balancebot_vrep_env_noise import BalanceBotVrepEnvNoise
from scripts.util import *
import numpy as np

class BalanceBotVrepEnvBalance(BalanceBotVrepEnvNoise):
	def compute_reward(self):
		'''Calculates the dense reward for the environment and adds a sparse reward for achieving the goal
		
		If the goal has been achieved a new goal is sampled.

		:returns: reward (sparse) + reward (goal_achievement)
		'''
		dense_reward = super().compute_reward()

		sparse_reward = 0.0
		if self.validate_goal():
			sparse_reward = 5.0
			self.goal = self.sample_goal()
			theta_continuous = self.observation[7:9]
			current_position = self.observation[9:11]	
			self.goal = np.array([	self.goal[1] * theta_continuous[1] + self.goal[0] * theta_continuous[0] + current_position[0], \
									self.goal[1] * theta_continuous[0] - self.goal[0] * theta_continuous[1] + current_position[1]])

		return dense_reward + sparse_reward

	def validate_goal(self):
		''' Check if the goal has been reached and maintained for X amount of time.

		:returns: boolean, based on wether the goal has been achieved or not.=
		'''
		
		time_till_goal_achieved = 1 # Seconds  
		goal_threshold = 0.1 # dist. in meters

		if self.observation[-1] < goal_threshold:
			self.steps += 1
		else: 
			self.steps = 0

		if self.steps > (time_till_goal_achieved * self.sample_rate):
			self.steps = 0
			print("--------------- Goal achieved --------------")
			return True
		else:
			return False 

	def compute_action(self, action):
		''' Transform the action vector 
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