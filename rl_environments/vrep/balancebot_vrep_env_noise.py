# TODO: Sparse goal rewards with goal thresholding and new goal definition
# The new goal environment is defined in more detail in Note-book. page 6,7
# TODO: Change the environment success condition
# TODO: Add a function that constructs the goal disc in v-rep so the behavior can be visually followed.
# TODO: Get a V-REP dongle so you can change the environment.

VECTOR_ACTION = True

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

from scripts.util import *

# #modify: the env class name
class BalanceBotVrepEnvNoise(vrep_env.VrepEnv):
	metadata = {
		'render.modes': [],
	}

	def __init__(
		self,
		server_addr='127.0.0.1',
		server_port=19997,
		scene_path=vrep_scenes_path+'/balance_test.ttt',
		goal_mode_on = True,
		verbose = True,
		goal_in_robot_frame = False
	):
		self.verbose = verbose
		self.goal_mode_on = goal_mode_on
		self.goal_in_robot_frame = True

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
		num_obs = 13
		if self.goal_mode_on:
			num_obs += 3
		
		# #modify: action_space and observation_space to suit your needs
		self.joints_max_velocity = 6.0
		act = np.array( [self.joints_max_velocity] * num_act )
		obs = np.array(		  [np.inf]		  * num_obs )
		#TODO: Change the observation space to reflect the actual boundaries of observation
		self.action_space	  = spaces.Box(-act,act)
		self.observation_space = spaces.Box(-obs,obs)
		self.goal_max = 2
		self.goal_space = spaces.Box(np.array([-self.goal_max,-self.goal_max]), np.array([self.goal_max,self.goal_max]))
		
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
		#Retrieve the pose from world frame to robot base
		pos = self.obj_get_position(self.oh_shape[0])
		
		orient = self.obj_get_orientation(self.oh_shape[0])
		#absolute yaw, part of the odom information
		abs_yaw = orient[2]
		#Construct Transformation matrix from world to robot base frame
		world_to_robot_rotation = transform_matrix(orient, pos)[0:3,0:3]

		#Add IMU data, Accel 3dof and Gyros 3dof
		# Observation dict keys: ['observation', 'achieved_goal', 'desired_goal']
		# IMU:
		# X.. Y.. Z.. 				: observation
		#TODO: Set the variables in the correct scripts in V-REP. I need a dongle to save these changes.
		#ax = self.get_float_signal('accelX')
		#ay = self.get_float_signal('accelY')
		#az = self.get_float_signal('accelZ')		
		#accel = [ax, ay, az]
		#gx = self.get_float_signal('gyroX')
		#gy = self.get_float_signal('gyroY')
		#gz = self.get_float_signal('gyroZ')
		#gyro = [gx, gy, gz]
	
		#Because i can't use the accelerometer yet i'll use lin_velocity deltas for accel:
		try:
			self.lin_vel_old = lin_vel
		except:
			self.lin_vel_old = np.array([0., 0., 0.])
		#print('Acceleration: {} Gyroscope: {}'.format(accel, gyro))
		# roll. yaw. 				: observation
		self.lin_vel, ang_vel = self.obj_get_velocity(self.oh_shape[0])
		#Rotate the velocity vectors to be represented in the robot base frame
		self.lin_vel = np.asarray(world_to_robot_rotation * np.matrix(self.lin_vel).T).reshape(-1)
		lin_acc = (self.lin_vel_old - self.lin_vel) * 20. #Multiply by 20 as there are 20 steps per second
		#Add the gravity vector to the lin_acceleration
		gravity_vector = np.array([0., 0., -9.81])
		grav_vec_base_frame = np.asarray(world_to_robot_rotation * np.matrix(gravity_vector).T).reshape(-1)
		lin_acc += grav_vec_base_frame
		####################################################
		
		#ang_vel = np.asarray(world_to_robot_rotation * np.matrix(ang_vel).T).reshape(-1)
		#orient  = np.asarray(world_to_robot_rotation * np.matrix(orient).T).reshape(-1)
		
		# L-Wheel-vel, R-wheel-vel	: observation
		try:
		 	self.l_wheel_old = self.l_angle
		 	self.r_wheel_old = self.r_angle
		except:
			pass
		self.l_angle = self.obj_get_joint_angle(self.oh_joint[0])
		self.r_angle = self.obj_get_joint_angle(self.oh_joint[1])
		# Calculating the delta joint angles using complex numbers to avoid the large delta jump at pi and -pi
		self.l_wheel_delta = np.complex(np.cos(self.l_angle), np.sin(self.l_angle)) \
							/np.complex(np.cos(self.l_wheel_old), np.sin(self.l_wheel_old))
		self.r_wheel_delta = np.complex(np.cos(self.r_angle), np.sin(self.r_angle)) \
							/np.complex(np.cos(self.r_wheel_old), np.sin(self.r_wheel_old))
		# Converting the complex deltas into angles
		self.l_wheel_delta = np.angle(self.l_wheel_delta)
		self.r_wheel_delta = np.angle(self.r_wheel_delta)
		
		self.observation = np.array([lin_acc[0], lin_acc[1], lin_acc[2], 
									 ang_vel[0], ang_vel[1], ang_vel[2], 
									 orient[0], np.cos(abs_yaw), np.sin(abs_yaw), 
									 pos[0], pos[1], 
									 self.r_wheel_delta, self.l_wheel_delta])
		self.add_sensor_noise

		if self.goal_mode_on:
			#Calculate the relative position vector
			relative_goal = np.complex(self.goal[0]-self.observation[9], self.goal[1]-self.observation[10])
			if self.goal_in_robot_frame:
				#Rotate the position vector into the robot frame
				rotated_vector = relative_goal * np.complex(self.observation[7], -self.observation[8])
				#Make the goal vector non-complex
				relative_goal = np.array([np.real(rotated_vector), np.imag(rotated_vector)])
			else:
				#Make the goal vector non-complex
				relative_goal = np.array([np.real(relative_goal), np.imag(relative_goal)])

			#Calculate the goal distance
			goal_dist = np.linalg.norm(relative_goal)
			self.observation = np.hstack([self.observation, relative_goal, goal_dist])

	def add_sensor_noise(self):
		for index in range(len(self.observation)):
			self.observation[index] += np.random.normal(0,0.05)
		
	
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
		self.steps += 1
        # transform the action from vector (lin and rot action) to motor control
		if VECTOR_ACTION:
			kinematics = np.matrix([[-1., 1.], [1., 1.]]) 
			#normalize_action = lambda x: np.asarray( ( x * 1./np.linalg.norm(x) * self.joints_max_velocity if np.linalg.norm(x) < )  )
			action = np.asarray(np.matrix(action) * kinematics).reshape(-1)
			#print(action)
			
		# #modify Either clip the actions outside the space or assert the space contains them
		action = np.clip(action,-self.joints_max_velocity, self.joints_max_velocity)
		#assert self.action_space.contains(action), "Action {} ({}) is invalid".format(action, type(action))
		
		# Actuate
		self._make_action(action)
		# Step
		self.step_simulation()
		# Observe
		self._make_observation()
		
		# Reward
		reward = self.compute_reward(action)
		if self.verbose:
			print('Reward: {0:1.4f}'.format(reward))
		# Check if the balancebot fell over 
		angle_base = self.obj_get_orientation(self.oh_shape[0])
		# Early stop
		tolerable_threshold = 0.707  #rads
		done = (np.abs(angle_base[0]) > tolerable_threshold or np.abs(angle_base[1]) > tolerable_threshold)
		# done = False
		
		return self.observation, reward, done, {}
	
	def compute_reward(self, action, achieved_goal=None, desired_goal=None, info=None):
		''' This function takes in observations, actions and goals and outputs reward
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
		r_regul = gaussian(delta_pos, sig=2.0)
		
		# The actual reward function is below, a standard format is being used to ensure the size of the reward remains predictable:
		# y_reward := (w_1 * a + w_2 * b + w_3 * c + ....) / (sum(w_1, w_2, w_3 ....)
		# a, b, c are different attributes that provide a reward, they do this on a scale no larger than 1
		# w_x is the weight for each attribute this provides the priority to different learned attributes
		# The sum of weights at the end is used to ensure that the max reward that can be recieved is 1.0
		r_alive = 1.0
		w = [10., 4., 0.]
		scale_factor = 1./sum(w)
		return (w[0] * r_alive + w[1] * (1. - norm_pos_dist) + w[2] * r_regul )* scale_factor

	def reset(self):
		"""Gym environment 'reset'
		"""
		if self.sim_running:
			self.stop_simulation()
		self.start_simulation()
		
		self.steps = 0
		#Unifrom pitch randomization, changing initial starting position 
		start_pitch = np.random.uniform(-np.pi/24, np.pi/24)
		self.obj_set_orientation(handle=self.oh_shape[0], eulerAngles=np.array([start_pitch, 0.0, 0.0]))
		
		self.pitch_offset = np.random.uniform(0,0.05)

		self.goal = self.sample_goal()
		print("Goal: ", self.goal)
		self._make_observation()
		return self.observation

	def sample_goal(self):
		goal = self.goal_space.sample()
		#goal = np.asarray([0.0, goal[0]])
		return goal
	
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
	   Agent takes random actions using 'env.action_space.sample()'
	"""
	# #modify: the env class name
	env = BalanceBotVrepEnvNoise()
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
