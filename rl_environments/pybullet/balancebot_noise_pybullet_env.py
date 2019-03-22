import logging
import numpy as np

import pybullet as p

from rl_environments.pybullet.balancebot_pybullet_env import BalanceBotPyBulletEnv

logger = logging.getLogger(__name__)

class BalanceBotNoisePyBulletEnv(BalanceBotPyBulletEnv):

    def _compute_observation(self):
        observation = super(BalanceBotPyBulletEnv, self)._compute_observation()
        return np.array([observation[0] + np.random.normal(0,0.05) + self.pitch_offset,
                observation[1] + np.random.normal(0,0.01),
                observation[2] + np.random.normal(0,0.05)])

    def _reset(self):
        self.pitch_offset = np.random.normal(0,0.1)
        observation = super(BalanceBotPyBulletEnv, self)._reset()
        return np.array([observation[0] + np.random.normal(0,0.05) + self.pitch_offset,
                observation[1] + np.random.normal(0,0.01),
                observation[2] + np.random.normal(0,0.05)])
