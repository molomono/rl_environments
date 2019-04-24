##################### Original code written by Yconst, https://github.com/yconst/balance-bot
## Accompanying blogpost: https://backyardrobotics.eu/2017/11/27/build-a-balancing-bot-with-openai-gym-pt-i-setting-up/
## 

import logging
import numpy as np

import pybullet as p

from rl_environments.pybullet.balancebot_pybullet_env_continuous import BalanceBotPyBulletEnvContinuous

logger = logging.getLogger(__name__)

class BalanceBotNoisePyBulletEnvContinuous(BalanceBotPyBulletEnvContinuous):

    def _compute_observation(self):
        observation = super(BalanceBotPyBulletEnvContinuous, self)._compute_observation()
        return np.array([observation[0] + np.random.normal(0,0.05) + self.pitch_offset,
                observation[1] + np.random.normal(0,0.01),
                observation[2] + np.random.normal(0,0.05)])

    def _reset(self):
        self.pitch_offset = np.random.normal(0,0.1)
        observation = super(BalanceBotPyBulletEnvContinuous, self)._reset()
        return np.array([observation[0] + np.random.normal(0,0.05) + self.pitch_offset,
                observation[1] + np.random.normal(0,0.01),
                observation[2] + np.random.normal(0,0.05)])
