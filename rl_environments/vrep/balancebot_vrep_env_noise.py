
# This file is a template for V-rep environments
#     all names in this file are just examples
# Search for '#modify' and replace accordingly


from rl_environments.vrep.balancebot_vrep_env import BalanceBotVrepEnv
import os
if os.name == 'nt':
	#print('If you are running this code on windows you need to manually define the vrep scene path in each respective environment.')
	vrep_scenes_path = 'C:\Program Files\V-REP3\V-REP_PRO\scenes'
else:
	vrep_scenes_path = os.environ['VREP_SCENES_PATH']

import gym
from gym import spaces

import numpy as np


class BalanceBotVrepEnvNoise(BalanceBotVrepEnv):

    def _make_observation(self):
        super(BalanceBotVrepEnv, self)._make_observation()
        self.observation = np.array([self.observation[0] + np.random.normal(0,0.05) + self.pitch_offset,
                self.observation[1] + np.random.normal(0,0.05),
                self.observation[2] + np.random.normal(0,0.05)])


    def reset(self):
        self.pitch_offset = np.random.normal(0,0.1)
        super(BalanceBotVrepEnv, self).reset()
        self.observation = np.array([self.observation[0] + np.random.normal(0,0.05) + self.pitch_offset,
                self.observation[1] + np.random.normal(0,0.05),
                self.observation[2] + np.random.normal(0,0.05)])
