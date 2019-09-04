import os
if os.name == 'nt':
	#print('If you are running this code on windows you need to manually define the vrep scene path in each respective environment.')
	vrep_scenes_path = 'C:/Program Files/V-REP3/V-REP_PRO/scenes'
else:
	vrep_scenes_path = os.environ['VREP_SCENES_PATH']

from .balancebot_vrep_env_noise import BalanceBotVrepEnvNoise
from scripts.util import *
import numpy as np

from vrep_env import vrep_env
from vrep_env import vrep # vrep.sim_handle_parent


class BalanceBotVrepEnvRotation(BalanceBotVrepEnvNoise):
	def reset(self):
		"""Rotation Reset function
		"""
		if self.sim_running:
			self.stop_simulation()
		self.start_simulation()
		
		self.steps = 0
		#Unifrom pitch randomization, changing initial starting position 
		start_pitch = np.random.uniform(-np.pi/18, np.pi/18)
		self.obj_set_orientation(handle=self.oh_shape[0], eulerAngles=np.array([start_pitch, 0.0, 0.0]))
		

		self.goal = self.sample_goal()
		print("Goal: ", self.goal)
		self._make_observation()
		return self.observation

	def compute_reward(self):
		'''Calculates the dense reward for the environment and adds a sparse reward for achieving the goal
		
		If the goal has been achieved a new goal is sampled.

		:returns: reward (sparse) + reward (goal_achievement)
		'''
		# Calculate the goal vector relative to the position of the balance-bot
		rel_pos_dist = np.array([self.goal[0]-self.observation[9], self.goal[1]-self.observation[10]])
		# Calculate the Angle of the goal with respect to the Y axis of the robot.
		# First calculate the angle of the goal with respect to the inertial Y axis
		goal_angle = np.array(	np.sin(np.arctan2(rel_pos_dist[1],rel_pos_dist[0])), \
								np.cos(np.arctan2(rel_pos_dist[1],rel_pos_dist[0])))
		# Retrieve the angle between the robot and the inertial X axis
		robot_angle = np.array(self.observation[7], self.observation[8])

		# Absolute Dot product, ranges from 0 to 1 rewarding alligning the Y axis of the robot with the goal.
		dense_reward = np.abs(np.dot(goal_angle, robot_angle))
		print("DENSE REWARD: ", dense_reward)
		sparse_reward = 0.0
		if self.validate_goal():
			sparse_reward = 50.0
			self.goal = self.sample_goal()

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
	
	def sample_goal(self):
		''' Samples the goal space for an XY coordinates

		:returns: numpy float array 
		'''
		goal = self.goal_space.sample()
		return goal