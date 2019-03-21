from gym.envs.registration import registry, register, make, spec

# Vrep
# ----------------------------------------
register(
    id='VrepCartPole-v0', 
    entry_point='rl_environments.vrep.cartpole_vrep_env:CartPoleVrepEnv', 		
    max_episode_steps=200, 
    reward_threshold=195.0
)
register(
    id='VrepCartPoleContinuous-v0', 
    entry_point='rl_environments.vrep.cartpole_continuous_vrep_env:CartPoleContinuousVrepEnv', 
    max_episode_steps=200, 
    reward_threshold=195.0
)
register(
    id='VrepBalanceBot-v0', 
    entry_point='rl_environments.vrep.balancebot_vrep_env:BalanceBotVrepEnv', 
    max_episode_steps=10000, 
    reward_threshold=100000.0
)
register(
    id='VrepHopper-v0', 
    entry_point='rl_environments.vrep.hopper_vrep_env:HopperVrepEnv', 
    max_episode_steps=1000
)
register(
    id='VrepDoubleCartPoleSwingup-v0', 
    entry_point='rl_environments.vrep.cartpole_continuous_swingup_vrep_env:DoubleCartPoleSwingupVrepEnv', 
    max_episode_steps=500
)