#TODO: This environment locks the pitch and roll of the robot allowing differential 
# drive to be used to maneuver to goal positions
# A reward is provided proportional to the distance the robot is from the goal.
# Once a goal position has been reached the robot must stay close before recieving an additional
# sparse reward.
# After the reward is provided a new goal is generated
# If the robot has not reached a goal in x seconds the environment is reset
# 
#


import os
if os.name == 'nt':
	#print('If you are running this code on windows you need to manually define the vrep scene path in each respective environment.')
	vrep_scenes_path = 'C:/Program Files/V-REP3/V-REP_PRO/scenes'
else:
	vrep_scenes_path = os.environ['VREP_SCENES_PATH']


from .balancebot_vrep_env_noise import BalanceBotVrepEnvNoise
from scripts.util import *
import numpy as np

class BalanceBotVrepEnvLocomotion(BalanceBotVrepEnvNoise):
	def reset(self):
		"""Gym environment 'reset'
		"""
		if self.sim_running:
			self.stop_simulation()
		self.start_simulation()
		
		self.steps = 0
		#Unifrom pitch randomization, changing initial starting position 
		start_pitch = np.random.uniform(-np.pi/18, np.pi/18)
		self.obj_set_orientation(handle=self.oh_shape[0], eulerAngles=np.array([start_pitch, 0.0, 0.0]))
		
		self.pitch_offset = np.random.uniform(0,0.05)

		self.goal = self.sample_goal()
		print("Goal: ", self.goal)
		self._make_observation()
		return self.observation



	def compute_reward(self, action, achieved_goal=None, desired_goal=None, info=None):
		pass
	
	def compute_action(self, action):
		pass