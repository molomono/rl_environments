###########
###########---- Big TODO: Stop the simulation environment when exiting the final simulation iteration
###########---- env.env.close()  or env.close() can be used to reset the scene, stop the simulation and disconnect TCP. 
###########---- This is necisarry other wise i need to manually pause the simulation before i can connect using a new AI.
###########
# This file is a template for V-rep environments
#	 all names in this file are just examples
# Search for '#modify' and replace accordingly


#################### WIP environment #######################
#TODO: Need to alter the observation space to contain observation, achieved goal and desired goal
#TODO: add compute_reward function, which is a function of achieved_goal vs desired_goal.
############################################################

'''
class GoalEnv(Env):
    """A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    def reset(self):
        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error('GoalEnv requires an observation space of type gym.spaces.Dict')
        for key in ['observation', 'achieved_goal', 'desired_goal']:
            if key not in self.observation_space.spaces:
                raise error.Error('GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(key))

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on an a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in info and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        """
        raise NotImplementedError
'''
VECTOR_ACTION = True

from vrep_env import vrep_env
from vrep_env import vrep # vrep.sim_handle_parent

from abstract_classes.sensor_info import SensorInfo

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
	return np.exp(-np.power(sig*np.sum(np.abs(x)),2.0))

def trig(r):
	return np.cos(r), np.sin(r)

def transform_matrix(rotation, translation):
	''' Returns homogeneous affine transformation matrix for a translation and rotation vector
	:param rotation: Roll, Pitch, Yaw as a 1d numpy.array or list
	:param translation: X, Y, Z as 1d numpy.array or list
	:returns: 3D Homogenous transform matrix 
	'''
	xC, xS = trig(rotation[0])
	yC, yS = trig(rotation[1])
	zC, zS = trig(rotation[2])
	dX = translation[0]
	dY = translation[1]
	dZ = translation[2]
	Translate_matrix =np.array([[1, 0, 0, dX],
								[0, 1, 0, dY],
								[0, 0, 1, dZ],
								[0, 0, 0, 1]])
	Rotate_X_matrix = np.array([[1, 0, 0, 0],
								[0, xC, -xS, 0],
								[0, xS, xC, 0],
								[0, 0, 0, 1]])
	Rotate_Y_matrix = np.array([[yC, 0, yS, 0],
								[0, 1, 0, 0],
								[-yS, 0, yC, 0],
								[0, 0, 0, 1]])
	Rotate_Z_matrix = np.array([[zC, -zS, 0, 0],
								[zS, zC, 0, 0],
								[0, 0, 1, 0],
								[0, 0, 0, 1]])
	return np.dot(Rotate_Z_matrix,np.dot(Rotate_Y_matrix,np.dot(Rotate_X_matrix,Translate_matrix)))

# #modify: the env class name
class BalanceBotVrepEnvNoise(vrep_env.VrepEnv, SensorInfo):
	metadata = {
		'render.modes': [],
	}

	def __init__(
		self,
		server_addr='127.0.0.1',
		server_port=19997,
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
		num_obs = 9
		
		# #modify: action_space and observation_space to suit your needs
		self.joints_max_velocity = 6.0
		act = np.array( [self.joints_max_velocity] * num_act )
		obs = np.array(		  [np.inf]		  * num_obs )
		#TODO: Change the observation space to reflect the actual boundaries of observation
		self.action_space	  = spaces.Box(-act,act)
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
		#Retrieve the pose from world frame to robot base
		pos = self.obj_get_position(self.oh_shape[0])
		orient = self.obj_get_orientation(self.oh_shape[0])
		#absolute yaw, part of the odom information
		abs_yaw = orient[2]
		#Construct Transformation matrix from world to robot base frame
		world_to_robot_transform = transform_matrix(pos, orient)
		#Extract the Rotation matrix from the transform matrix
		world_to_robot_rotation = world_to_robot_transform[0:3,0:3] 

		#Add IMU data, Accel 3dof and Gyros 3dof
		# Observation dict keys: ['observation', 'achieved_goal', 'desired_goal']
		# IMU:
		# X.. Y.. Z.. 				: observation
		#force, torque = self.obj_read_force_sensor(self.forcesensor)
		#accel = np.asarray(force) / 1.000e-3
		#print('Acceleration',accel)
		# roll. yaw. 				: observation
		lin_vel, ang_vel = self.obj_get_velocity(self.oh_shape[0])
		#Rotate the velocity vectors to be represented in the robot base frame
		lin_vel = world_to_robot_rotation * np.matrix(lin_vel).T
		ang_vel = world_to_robot_rotation * np.matrix(ang_vel).T

		# L-Wheel-vel, R-wheel-vel	: observation
		try:
		 	self.l_wheel_old = self.l_angle
		 	self.r_wheel_old = self.r_angle
		except:
			pass
		self.l_angle = self.obj_get_joint_angle(self.oh_joint[0])
		self.r_angle = self.obj_get_joint_angle(self.oh_joint[1])
		self.l_wheel_delta = self.l_angle - self.l_wheel_old
		self.r_wheel_delta = self.r_angle - self.r_wheel_old
		
		self.observation = np.array([ang_vel[0], ang_vel[2], np.cos(abs_yaw), np.sin(abs_yaw), ang_vel[1], pos[0], pos[1], self.r_wheel_delta, self.l_wheel_delta])
		
	def add_sensor_noise(self):
		#mean, covar = sensors.get_sensor_noise_params()
    	#self.observation = self.observation + np.random.normal( mean, covar)
		self.observation = np.array([self.observation[0] + np.random.normal(0,0.05),
					self.observation[1] + np.random.normal(0,0.05),
					self.observation[2] + np.random.normal(0,0.05),
					self.observation[3] + np.random.normal(0,0.05),
					self.observation[4] + np.random.normal(0,0.05),
					self.observation[5] + np.random.normal(0,0.05),
					self.observation[6] + np.random.normal(0,0.05),
					self.observation[7] + np.random.normal(0,0.05),
					self.observation[8] + np.random.normal(0,0.05),
					self.observation[9] + np.random.normal(0,0.05)])
	
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
        # transform the action from vector (lin and rot action) to motor control
		if VECTOR_ACTION:
			kinematics = np.matrix([[0., 0.], [1., 1.]]) 
			#normalize_action = lambda x: np.asarray( ( x * 1./np.linalg.norm(x) * self.joints_max_velocity if np.linalg.norm(x) < )  )
			action = np.asarray(np.matrix(action) * kinematics).reshape(-1)
			print(action)
			
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
		
		#Check if the balancebot fell over 
		angle_base = self.obj_get_orientation(self.oh_shape[0])
		# Early stop
		tolerable_threshold = np.pi*2/3  #rads
		done = (np.abs(angle_base[0]) > tolerable_threshold or np.abs(angle_base[1]) > tolerable_threshold)
		#done = False
		
		return self.observation, reward, done, {}
	
	def compute_reward(self, action, achieved_goal=None, desired_goal=None, info=None):
		''' This function takes in observations, actions and goals and outputs reward
		'''
        #modify the reward computation
		# example: possible variables used in reward
		head_pos_x = self.observation[4] # front/back
		head_pos_y = self.observation[5] # left/right
		theta  	= gaussian( self.observation[3], sig=2.5 ) 

		norm_pos_dist = 1-(np.linalg.norm([head_pos_x,head_pos_y]) * 1./np.linalg.norm([10,10]))

		delta_pos = np.asarray([self.l_wheel_delta, self.r_wheel_delta])
		r_regul = gaussian(delta_pos, sig=2.0)
		
		print("regulation factors, wheel: {}, pitch: {}, pos_dist: {}".format(r_regul, theta, norm_pos_dist))
		##
		r_alive = 1.0
		a = 1./12.
		return (8.*r_alive + theta + r_regul + 2.*norm_pos_dist) * a

	def reset(self):
		"""Gym environment 'reset'
		"""
		if self.sim_running:
			self.stop_simulation()
		self.start_simulation()
		
		#Unifrom pitch randomization, changing initial starting position 
		start_pitch = np.random.uniform(-np.pi/24, np.pi/24)
		self.obj_set_orientation(handle=self.oh_shape[0], eulerAngles=np.array([start_pitch, 0.0, 0.0]))
		
		self.pitch_offset = np.random.uniform(0,0.05)

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
