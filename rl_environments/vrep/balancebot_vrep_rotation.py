import os

from .balancebot_vrep_env_noise import BalanceBotVrepEnvNoise
from scripts.util import *
import numpy as np

from vrep_env import vrep_env
from vrep_env import vrep # vrep.sim_handle_parent


class BalanceBotVrepEnvRotation(BalanceBotVrepEnvNoise):
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
		# goal_position in robot frame
		goal_position_robot =  np.complex(self.observation[7], self.observation[8]).conjugate() * np.complex(self.observation[-3], self.observation[-2])
		# Dense reward in principle is abs(Y_relative / || goal_relative ||)
		# In other words, the alignment of the Y-axis of the robot with the goal
		dense_reward = 2.*np.abs(goal_position_robot.imag / np.linalg.norm(self.observation[-1])) -1
		sparse_reward = 0.0
		if self.validate_goal(dense_reward):
			sparse_reward = 100.
			self.goal = self.sample_goal()

		return dense_reward + sparse_reward

	def validate_goal(self,dense_reward):
		''' Check if the goal has been reached and maintained for X amount of time.

		:returns: boolean, based on wether the goal has been achieved or not.=
		'''
		
		time_till_goal_achieved = 1 # Seconds  
		goal_threshold = 0.025 # threshold accuracy, sin( pi/128 [rads])

		#check if the abs distance to the goal is smaller than the theshold
		if dense_reward > (1.-goal_threshold):
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