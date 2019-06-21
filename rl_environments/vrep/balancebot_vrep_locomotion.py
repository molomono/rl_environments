#TODO: This environment locks the pitch and roll of the robot allowing differential 
# drive to be used to maneuver to goal positions
# A reward is provided proportional to the distance the robot is from the goal.
# Once a goal position has been reached the robot must stay close before recieving an additional
# sparse reward.
# After the reward is provided a new goal is generated
# If the robot has not reached a goal in x seconds the environment is reset
# 
#