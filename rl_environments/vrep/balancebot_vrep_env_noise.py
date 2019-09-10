# TODO: Add a function that constructs the goal disc in v-rep so the behavior can be visually followed.

# TODO: Add domain and dynamics randomization

from vrep_env import vrep_env
from vrep_env import vrep # vrep.sim_handle_parent

import os
if os.name == 'nt':
	#print('If you are running this code on windows you need to manually define the vrep scene path in each respective environment.')
	vrep_scenes_path = 'C:/Program Files/V-REP3/V-REP_PRO/scenes'
else:
	vrep_scenes_path = os.environ['VREP_SCENES_PATH']

import gym
from gym import spaces

import numpy as np

from scripts.util import *

class BalanceBotVrepEnvNoise(vrep_env.VrepEnv):
	metadata = {
		'render.modes': [],
	}

	def __init__(
		self,
		server_addr='127.0.0.1',
		server_port=19997,
		scene_path=vrep_scenes_path+'/balance_bot_ram.ttt'
	):
		''' Initialization of the robot environment.

		Provided a V-REP server is being hosted on the server address and port provided.
		This function connects to V-REP, loads the scene from the scene path.
		It construcst the handles used to communicate with V-REP and subsequently
		defines the action and observation spaces.


		:param server_addr: The server IP address the V-REP server is hosted on
		:param server_port: The port the server is hosted on
		:param scene_path: The path to the scene file, this location is local to the V-REP server.
		'''
		self.verbose = True
		self.goal_mode_on = True
		self.goal_in_robot_frame = False
		self.sample_rate = 20
		self.vector_action = True

		vrep_env.VrepEnv.__init__(self,server_addr,server_port,scene_path)
		# List of joint names, which match with the joints in the V-REP scene
		joint_names = ['l_wheel_joint','r_wheel_joint']
		# List of object names, which match with the jonits in the V-REP scene
		shape_names = ['base', 'l_wheel', 'r_wheel']

		# Get object handle IDs
		# Actuators
		self.oh_joint = list(map(self.get_object_handle, joint_names))
		# Shapes
		self.oh_shape = list(map(self.get_object_handle, shape_names))
		# Goal
		self.oh_goal = list(map(self.get_object_handle, ['goal_object']))
		
		# The number of actions is equal to the number of joints, Left and right wheel
		num_act = len(self.oh_joint)
		
		# Define the dimensions of the observation space 
		num_obs = 14
		if self.goal_mode_on:
			num_obs += 3
		
		# Define minimum and maximum forces that actuators can apply
		self.min_torque = 0.0
		self.max_torque = 0.75 * 1.25
		# Define action and observation space
		self.joints_max_velocity = 1 #25 #max torque set in vrep
		act = np.array( [self.joints_max_velocity] * num_act )
		#The first 6 observations are bound to a range between -1 and 1
		obs = np.array( ([1.0] * 6) + ([np.inf] * (num_obs-6)) )

		#TODO: Change the observation space to reflect the actual boundaries of observation
		self.action_space	  = spaces.Box(-1.0*act,act)
		self.observation_space = spaces.Box(-1.0*obs,obs)
		self.goal_max = 4
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
		# IMU:
		# X.. Y.. Z.. 				: observation

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

		# roll. yaw. 				: observation
		self.lin_vel, ang_vel = self.obj_get_velocity(self.oh_shape[0])
		#Rotate the velocity vectors to be represented in the robot base frame
		self.lin_vel = np.asarray(world_to_robot_rotation * np.matrix(self.lin_vel).T).reshape(-1)
		lin_acc = (self.lin_vel_old - self.lin_vel) * 20. #Multiply by 20 as there are 20 steps per second
		#Add the gravity vector to the lin_acceleration
		gravity_vector = np.array([0., 0., -9.81])
		grav_vec_base_frame = np.asarray(world_to_robot_rotation * np.matrix(gravity_vector).T).reshape(-1)
		lin_acc += grav_vec_base_frame
	
		# L-Wheel-vel, R-wheel-vel	: observation
		self.l_angle = self.obj_get_joint_angle(self.oh_joint[0])
		self.r_angle = self.obj_get_joint_angle(self.oh_joint[1])
		# Calculating the delta joint angles using complex numbers to avoid the large delta jump at pi and -pi
		self.l_wheel_delta = np.complex(np.cos(self.l_angle), np.sin(self.l_angle)) \
							/np.complex(np.cos(self.l_wheel_old), np.sin(self.l_wheel_old))
		self.r_wheel_delta = np.complex(np.cos(self.r_angle), np.sin(self.r_angle)) \
							/np.complex(np.cos(self.r_wheel_old), np.sin(self.r_wheel_old))
		# Converting the complex deltas into angles
		self.l_wheel_delta = np.angle(self.l_wheel_delta) * self.sample_rate
		self.r_wheel_delta = np.angle(self.r_wheel_delta) * self.sample_rate
		
		lin_vel = (self.r_wheel_delta + self.l_wheel_delta) / 2.
		self.observation = np.array([lin_acc[0], lin_acc[1], lin_acc[2], 
									 ang_vel[0], ang_vel[1], ang_vel[2], 
									 orient[0], np.cos(abs_yaw), np.sin(abs_yaw), 
									 pos[0], pos[1], 
									 self.r_wheel_delta, self.l_wheel_delta,
									 lin_vel])

		self.remap_observations()
		self.add_sensor_noise()

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

	def remap_observations(self):
		'''Linear remapping of the input range to a defined output range
		'''
		#Accel
		self.observation[0:3] = remap(self.observation[0:3], 2*-9.81, 2*9.81, -1, 1)
		#Gyro
		self.observation[3:6] = remap(self.observation[3:6], -8.727, 8.727, -1, 1)

		#self.observation[6] 		#Absolute pitch
		#self.observation[7:9]   	#Continuous theta
		#self.observation[9:11]		#Absolute Position
		#self.observation[11:13] 	#Wheel Velocity * 1/self.sample_rate
		#self.observation[13] 		#linear velocity


	def add_sensor_noise(self):
		#adding normal noise to the accelerometer 
		self.observation[0:3] = np.random.normal(self.observation[0:3], np.array([0.00675]*3))
		self.observation[3:6] = np.random.normal(self.observation[3:6], np.array([0.00250]*3))	
	
	def _make_action(self, a):
		"""Query V-rep to make action.
		   no return value
		"""
		# VREP handles torque control in an obscure mannor:
		# 1. Set unreachable Velocity in the direction of desired rotation: 
		# sign(action) * 10000  to set velocity
		# 2. Modulate the maximum Joint force to restrain the torque being applied
		# maximum torque is mapped from -1 to 1 --> 0 to 25 Nm
		for i_oh, i_a in zip(self.oh_joint, a):
			self.obj_set_velocity(i_oh, np.sign(i_a) * 1000.)
			self.obj_set_force(i_oh, remap(np.abs(i_a), self.action_space.low[0], self.action_space.high[0], self.min_torque, self.max_torque))
			#vrep.simxSetJointForce(self.cID, i_oh, i_a, vrep.simx_opmode_continuous)
	
	def compute_action(self, action):
		''' Transform the action vector 

		The predicted action can be a rotation and translation vector,
		these need to be transformed into power of the motors.
		
		:param action: predicted action
		:returns: Action in the form of ratio of motor power -1 to 1
		'''
		kinematics = np.matrix([[-1., 1.], [1., 1.]]) 
		return np.asarray(np.matrix(action) * kinematics).reshape(-1)

	def step(self, action):
		"""Gym environment 'step'
		
		"""
        # transform the action from vector (lin and rot action) to motor control
		if self.vector_action:
			action = self.compute_action(action)
			
		#clip the action to the correct range	
		action = np.clip(action, self.action_space.low, self.action_space.high)

		# Actuate
		self._make_action(action)
		# Step
		self.step_simulation()
		# Observe
		self._make_observation()
		
		# Reward
		reward = self.compute_reward()

		if self.verbose: # Print the reward and action if verbosity is turned on
			print('Reward: {0:1.4f}'.format(reward))
			print("clipped action: {}".format(action))
		# Check if the balancebot fell over 
		angle_base = self.obj_get_orientation(self.oh_shape[0])
		# Early stop
		tolerable_threshold = 0.707  #rads
		done = (np.abs(angle_base[0]) > tolerable_threshold or np.abs(angle_base[1]) > tolerable_threshold)
		# done = False
		self.visualize_goal_position(self.goal)
		return self.observation, reward, done, {}
	
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
		r_regul = np.sum(gaussian(delta_pos, sigma=2.0))
		print("R_regul: ",r_regul)
		# The actual reward function is below, a standard format is being used to ensure the size of the reward remains predictable:
		# y_reward := (w_1 * a + w_2 * b + w_3 * c + ....) / (sum(w_1, w_2, w_3 ....))
		# a, b, c are different attributes that provide a reward, they do this on a scale no larger than 1
		# w_x is the weight for each attribute this provides the priority to different learned attributes
		# The sum of weights at the end is used to ensure that the max reward that can be recieved is 1.0
		r_alive = 1.0
		w = [0., 4., 0.]
		scale_factor = 1./sum(w)
		return (w[0] * r_alive + w[1] * (1. - norm_pos_dist) + w[2] * r_regul )* scale_factor		
		#return (w[0] * r_alive + w[1] * (norm_pos_dist/norm_pos_dist * np.power((1. - norm_pos_dist),2) + w[2] * r_regul )* scale_factor
		
	def reset(self):
		"""Gym environment 'reset'

		Stops the simulation if it is still running. Subsequently the 
		environment is started anew. 

		:returns: observation vector
		"""
		if self.sim_running:
			self.stop_simulation()
		self.start_simulation()
		
		#reset the number of steps run in this rollout
		self.steps = 0
		#Unifrom pitch randomization, changing initial starting position 
		start_pitch = np.random.uniform(-np.pi/18, np.pi/18)
		self.obj_set_orientation(handle=self.oh_shape[0], eulerAngles=np.array([start_pitch, 0.0, 0.0]))
	
		#Sample an initial goal
		self.goal = self.sample_goal()
		print("Goal: ", self.goal)

		#Make an initial observation - used to take the first action
		self._make_observation()
		return self.observation

	def sample_goal(self):
		''' Samples the goal space for an XY coordinates

		:returns: numpy float array 
		'''
		goal = self.goal_space.sample() * 0.0
		return goal

	def visualize_goal_position(self, goal):
		#def obj_set_position(self, handle, pos, relative_to=None):
		#return self.RAPI_rc(vrep.simxSetObjectPosition( self.cID,handle,
		#	-1 if relative_to is None else relative_to,
		#	pos,
		#	self.opM_set))
		goal_position = np.hstack([goal, 0.01]) # x, y, 0.01 in [m]
		self.obj_set_position(self.oh_goal[0], goal_position)

	
def main(args):
	"""main function used as test and example.
	   Agent takes random actions using 'env.action_space.sample()'
	"""
	# #modify: the env class name
	env = BalanceBotVrepEnvNoise()
	for _ in range(8): 
		env.reset()
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
