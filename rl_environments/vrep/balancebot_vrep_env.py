
# This file is a template for V-rep environments
#     all names in this file are just examples
# Search for '#modify' and replace accordingly

from vrep_env import vrep_env
from vrep_env import vrep # vrep.sim_handle_parent

import os
if os.name == 'nt':
	#print('If you are running this code on windows you need to manually define the vrep scene path in each respective environment.')
	vrep_scenes_path = 'C:\Program Files\V-REP3\V-REP_PRO\scenes'
else:
	vrep_scenes_path = os.environ['VREP_SCENES_PATH']

import gym
from gym import spaces

import numpy as np

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def rads2complex(rads):
	return [np.sin(rads), np.cos(rads)]

def gaussian_2d(x,y, scale_x=0.5, scale_y=0.5):
    r = np.abs(np.add(scale_x*np.power(x,2), scale_y*np.power(y,2)))
    return 1.0 - np.tanh(1.0/(np.pi*2.0)*r)

def gaussian(x,sig=1.0):
    return np.exp(-np.power(sig*np.linalg.norm(x),2.0))

# #modify: the env class name
class BalanceBotVrepEnv(vrep_env.VrepEnv):
	metadata = {
		'render.modes': [],
	}
	def __init__(
		self,
		server_addr='127.0.0.1',
		server_port=19997,
		# #modify: the filename of your v-rep scene
		scene_path=vrep_scenes_path+'/balance_test.ttt'
	):
		
		vrep_env.VrepEnv.__init__(self,server_addr,server_port,scene_path)
		# #modify: the name of the joints to be used in action space
		joint_names = ['l_wheel_joint','r_wheel_joint']
		# #modify: the name of the shapes to be used in observation space
		shape_names = ['base', 'l_wheel', 'r_wheel']
		
		## Getting object handles
		# we will store V-rep object handles (oh = object handle)

		# Meta
		# #modify: if you want additional object handles
		self.camera = self.get_object_handle('camera')
		
		# Actuators
		self.oh_joint = list(map(self.get_object_handle, joint_names))
		# Shapes
		self.oh_shape = list(map(self.get_object_handle, shape_names))
		
		
		# #modify: if size of action space is different than number of joints
		# Example: One action per joint
		num_act = len(self.oh_joint)
		
		# #modify: if size of observation space is different than number of joints
		# Example: 3 dimensions of linear and angular (2) velocities + 6 additional dimension
		# 3 =  X, Y, Theta thus planar position (Might want to expand it to the velocities as well)
		#num_obs = 12 
		num_obs = 10
		# #modify: action_space and observation_space to suit your needs
		self.joints_max_velocity = 3.0
		act = np.array( [self.joints_max_velocity] * num_act )
		obs = np.array(          [np.inf]          * num_obs )
		#TODO: Change the observation space to reflect the actual boundaries of observation
		self.action_space      = spaces.Box(-act,act)
		self.observation_space = spaces.Box(-obs,obs)
		
		#the placeholders for the delta position of the wheel encoders
		# i should instead instantiate them during the first step using if var in locals:
		self.l_wheel_delta = 0.0
		self.r_wheel_delta = 0.0
		self.l_wheel_old = 0.0
		self.r_wheel_old = 0.0

		# #modify: optional message
		print('BalanceBot Environment: initialized')
	
	def _make_observation(self):
		"""Query V-rep to make observation.
		   The observation is stored in self.observation
		"""
		# start with empty list
		lst_o = []

		#Definition of the observation vector:
		# [x, y, theta, base_ang_vel x 3, wheel_encoder_l, wheel_encoder_r]
		pos = self.obj_get_position(self.oh_shape[0])
		ang_pos = self.obj_get_orientation(self.oh_shape[0])
		lin_vel, ang_vel = self.obj_get_velocity(self.oh_shape[0])
		self.ang_vel = ang_vel

		#calculate the relative velocity of the balance bot
		#TODO: change the way i fixed this, but becase the balance bot is differential drive and only translates in x
		#i'm acting like the relative velocity is the distance function between the two vectors, This does not hold true if the
		#robot slips, so i should transform it with a transform matrix instead, this is a quick fix for now.
		rel_lin_vel_x = np.sqrt( np.power(lin_vel[0], 2) + np.power(lin_vel[1], 2) )

		lst_o += pos[0:2]
		#Theta is in Radians, make it a complex number
		lst_o += [np.sin(ang_pos[2]), np.cos(ang_pos[2])]
		lst_o += ang_vel

		#add the lin velocity
		lst_o += [rel_lin_vel_x]
		try:
			self.l_wheel_old = self.l_angle
			self.r_wheel_old = self.r_angle
		except:
			pass
		#add wheel angles to the observation
		self.l_angle = self.obj_get_joint_angle(self.oh_joint[0])
		self.r_angle = self.obj_get_joint_angle(self.oh_joint[1])
		# I should calculate the wheel velocity in complex numbers
		#if self.r_angle in locals():
		self.l_wheel_delta = self.l_angle - self.l_wheel_old
		self.r_wheel_delta = self.r_angle - self.r_wheel_old
		lst_o += [self.l_wheel_delta, self.r_wheel_delta]
		#lst_o += [np.sin(l_angle), np.cos(l_angle)]
		#lst_o += [np.sin(r_angle), np.cos(r_angle)]

		self.observation = np.array(lst_o).astype('float32');
	
	def _make_action(self, a):
		"""Query V-rep to make action.
		   no return value
		"""
		# #modify
		# example: set a velocity for each joint
		for i_oh, i_a in zip(self.oh_joint, a):
			self.obj_set_velocity(i_oh, i_a)
	
	def step(self, action):
		"""Gym environment 'step'
		"""
		# #modify Either clip the actions outside the space or assert the space contains them
		# actions = np.clip(actions,-self.joints_max_velocity, self.joints_max_velocity)
		assert self.action_space.contains(action), "Action {} ({}) is invalid".format(action, type(action))
		
		# Actuate
		self._make_action(action)
		# Step
		self.step_simulation()
		# Observe
		self._make_observation()
		
		# Reward
		# #modify the reward computation
		# example: possible variables used in reward
		head_pos_x = self.observation[0] # front/back
		head_pos_y = self.observation[1] # left/right
		theta  	= gaussian( self.observation[2], sig=1.5 ) 

		#TODO: change the action to the deltaPos of the wheels:
		delta_pos = np.asarray([self.l_wheel_delta, self.r_wheel_delta])
		#print(delta_pos)
		r_regul = gaussian( 20* delta_pos, sig=1.0)
		r_alive = 2.0
		# example: different weights in reward 
		#attempts to stay alive and stay centered
		
		#Reward is formulated as reward = a(f(x)) + b
		#where f is the reward function and a and b are used to alter the
		#starting point and magnitude of the reward function
		#for certain types of RL such as DRL this can always be done
		#and can improve convergence properties
		a = 0.1
		b = -1.0		
		reward = (a*(5.0*(r_alive) + 0.1*r_regul) + b) * 10
		#reward = r_regul
		#TODO: The reward function punishes high action, however action is torque, THIS IS FIXED NOW
		# This seems to be bad because a change of velocity is what we want to control, 
		# it is rather the continual accumilation of kinetic energy that we want to diminish.
		# Therfore the accumilated VELOCITY of the weels should be punushed per time step. 
		#reward = a*( (r_alive + gaussian_2d(head_pos_x, head_pos_y)) - (r_regul)) + b
		#+ (1.0)* gaussian_2d(head_pos_x, head_pos_y) + (1.0)*theta
		
		#Check if the balancebot fell over 
		angle_base = self.obj_get_orientation(self.oh_shape[0])
		# Early stop
		tolerable_threshold = 0.5  #rads
		done = (np.abs(angle_base[0]) > tolerable_threshold or np.abs(angle_base[1]) > tolerable_threshold)
		#done = False
		
		return self.observation, reward, done, {}
	
	def reset(self):
		"""Gym environment 'reset'
		"""
		if self.sim_running:
			self.stop_simulation()
		self.start_simulation()
		self._make_observation()
		return self.observation
	
	def render(self, mode='human', close=False):
		"""Gym environment 'render'
		"""
		pass
	
	def seed(self, seed=None):
		"""Gym environment 'seed'
		"""
		return []
	
def main(args):
	"""main function used as test and example.
	   Agent does random actions with 'action_space.sample()'
	"""
	# #modify: the env class name
	env = BalanceBotVrepEnv()
	for i_episode in range(8):
		observation = env.reset()
		total_reward = 0
		for t in range(256):
			action = env.action_space.sample()
			observation, reward, done, _ = env.step(action)
			total_reward += reward
			if done:
				break
		print("Episode finished after {} timesteps.\tTotal reward: {}".format(t+1,total_reward))
	env.close()
	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
