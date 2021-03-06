import os

from .balancebot_vrep_env_noise import BalanceBotVrepEnvNoise
from scripts.util import *
import numpy as np

from vrep_env import vrep_env
from vrep_env import vrep # vrep.sim_handle_parent

class BalanceBotVrepEnvLocomotion(BalanceBotVrepEnvNoise):
	def reset(self):
		"""Locomotion Reset function
		"""
		self.steps = 0
		return super().reset()

	def compute_reward(self):
		'''Calculates the dense reward for the environment and adds a sparse reward for achieving the goal
		
		If the goal has been achieved a new goal is sampled.

		:returns: reward (sparse) + reward (goal_achievement)
		'''
		#Dense reward for distance from goal
		dense_reward = super().compute_reward()

		#Dense reward for angle with respect to goal
		goal_position_robot =  np.complex(self.observation[7], self.observation[8]).conjugate() * np.complex(self.observation[-3], self.observation[-2])
		# Dense reward in principle is abs(Y_relative / || goal_relative ||)
		# In other words, the alignment of the Y-axis of the robot with the goal
		rotation_reward = np.abs(goal_position_robot.imag / np.linalg.norm(self.observation[-1]))
		
		if self.observation[-1] < 0.5:
			rotation_reward = 0.5*rotation_reward + 0.5

		#Sparse reward for achieving a goal pose
		sparse_reward = 0.0
		if self.validate_goal():
			sparse_reward = 100.0
			self.goal = self.sample_goal()

		return dense_reward + 0.5 * rotation_reward + sparse_reward

	def validate_goal(self):
		''' Check if the goal has been reached and maintained for X amount of time.

		:returns: boolean, based on wether the goal has been achieved or not.=
		'''
		
		time_till_goal_achieved = 0.5 # Seconds  
		goal_threshold = 0.25 # dist. in meters

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
	
	def sample_goal(self):
		''' Samples the goal space for an XY coordinates

		:returns: numpy float array 
		'''
		goal = self.goal_space.sample()
		return goal
