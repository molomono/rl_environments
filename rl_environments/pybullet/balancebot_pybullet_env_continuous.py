##################### Original code written by Yconst, https://github.com/yconst/balance-bot ###################################
## Accompanying blogpost: https://backyardrobotics.eu/2017/11/27/build-a-balancing-bot-with-openai-gym-pt-i-setting-up/
## 
## Original code modified by Brian Armstrong
## Changes:
##      - Randomizing the Domain and Dynamics
##      - Making the action space continuous
##      - TODO: Changes to observation space
##      - TODO: Gaussian noise now models a real robot sensor noise (Loomo)
##      - TODO: Changes to the reward function
##      - TODO: New definition for completing the environment
##      - TODO: New Robot Model (Resembling the Loomo)
##
## NOTE:
## 1. The noise wrapper does not work yet, fix this
## 2. The environment cannot be rendered on ubuntu due to openGL
##      See if there is a possibility to remotely run it on windows and access it through sockets

import os
import math
import numpy as np


import gym
from gym import spaces
from gym.utils import seeding

import pybullet as p
import pybullet_data

#TODO: Conenct to the 

import yaml
#Move this function to a seperate importable file.
#Not sure this is the way to go, 
def load_yaml(file_path, io_type='r'):
    with open(file_path, io_type) as stream:
        try:
            return(yaml.load(stream))
        except yaml.YAMLError as exc:
            print(exc)

class BalanceBotPyBulletEnvContinuous(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, render=False):
        self._observation = []
        self.min_action = -5.0
        self.max_action =  5.0
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(2,))
        self.observation_space = spaces.Box(np.array([-math.pi, -math.pi, -math.pi, -np.inf, -np.inf]), 
                                            np.array([math.pi, math.pi, math.pi, np.inf, np.inf])) # pitch, gyro, com.sp.

        # if (render):
        #     self.physicsClient = p.connect(p.TCP, "localhost", 6667)
        # else:
        #     self.physicsClient = p.connect(p.DIRECT)  # non-graphical version

        self.physicsClient = p.connect(p.TCP, "localhost", 6667)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self._assign_throttle(action)
        p.stepSimulation()
        self._observation = self._compute_observation()
        reward = self._compute_reward()
        done = self._compute_done()

        self._envStepCounter += 1

        return np.array(self._observation), reward, done, {}

    def _reset(self):
        # reset is called once at initialization of simulation
        self.vt = 0
        self.vd = 0
        self.maxV = 24.6 # 235RPM = 24,609142453 rad/sec
        self._envStepCounter = 0

        #Load parameters for environment randomization from yaml
        
        #Reset simulation, randomize Domain and Dynamics
        #TODO: WIP WIP WIP
        #simulation_params = yaml.load("simulation_parameters.yaml")

        # Domain can be set in this file
        # Dynamics must be altered in the robot.xml/robot.urdf script
        p.resetSimulation()
        g = 9.81
        dt = 0.01

        p.setGravity(0,0,-g) # m/s^2
        p.setTimeStep(dt) # sec
        planeId = p.loadURDF("plane.urdf")
        cubeStartPos = [0,0,0.001]
        tilt = (np.random.rand(1)-0.5)
        yaw =  (np.random.rand(1)-0.5*np.pi)

        cubeStartOrientation = p.getQuaternionFromEuler([tilt,0.,0.])

        path = os.path.abspath(os.path.dirname(__file__))
        path = "C:/github/rl_environments/rl_environments/pybullet/"
        self.botId = p.loadURDF(path + "balancebot_simple.urdf",
                           cubeStartPos,
                           cubeStartOrientation) + 1
                           
        # you *have* to compute and return the observation from reset()
        self._observation = self._compute_observation()
        return np.array(self._observation)

    def _assign_throttle(self, action):
        dv = 0.1
        deltav = action*dv
        vt = np.clip(self.vt + deltav, -self.maxV, self.maxV)
        self.vt = vt

        p.setJointMotorControl2(bodyUniqueId=self.botId, 
                                jointIndex=0, 
                                controlMode=p.VELOCITY_CONTROL, 
                                targetVelocity=vt[0])
                                
        p.setJointMotorControl2(bodyUniqueId=self.botId, 
                                jointIndex=1, 
                                controlMode=p.VELOCITY_CONTROL, 
                                targetVelocity=-vt[1])

    def _compute_observation(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.botId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        linear, angular = p.getBaseVelocity(self.botId)
        return np.array(cubeEuler + angular[0:2])

    def _compute_reward(self):
        return 0.1 - np.mean(abs(self.vt - self.vd)) * 0.005

    def _compute_done(self):
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.botId)
        cubeEuler = p.getEulerFromQuaternion(cubeOrn)
        print(cubeEuler)
        return abs(cubeEuler[0]) > np.pi/3 or self._envStepCounter >= 1500

    def _render(self, mode='human', close=False):
        pass

#def clamp(n, minn, maxn):
#    return max(min(maxn, n), minn)
