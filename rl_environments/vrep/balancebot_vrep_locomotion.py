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

from vrep_env import vrep_env
from vrep_env import vrep # vrep.sim_handle_parent

class BalanceBotVrepEnvLocomotion(BalanceBotVrepEnvNoise):
	
	time_till_goal_achieved = 2  

	def reset(self):
		"""Locomotion Reset function
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
		#self.fix_orientation()
		return self.observation

	#def fix_orientation(self, eulerAngles):
	#	'''This function locks the pitch/roll of the balance bot making it impossible 
	#	for the robot to fall over.
	#	'''
	#	#def obj_set_orientation(self, handle, eulerAngles, relative_to=None):
	#	vrep.simxSetObjectOrientation( self.cID, self.oh_shape[0], -1, eulerAngles, vrep.simx_opmode_continuous)

	def compute_reward(self, action, achieved_goal=None, desired_goal=None, info=None):
		pass

	def validate_goal(self):
		print(self.sample_rate * self.time_till_goal_achieved)
		pass
	
	def compute_action(self, action):
		pass