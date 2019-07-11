.. Environments_docs documentation master file, created by
   sphinx-quickstart on Mon Jul  1 12:32:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The Demonstrator
================
This documentation page contains information regarding the Demonstrator project. 
The premis of the project is to design a self-learning algorithm with which a two-wheeled 
balance robot can learn to balance and perform simple locomotive tasks.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    envs/index
    agents/index 
    opt/index


Graph-viz Test
==============
This section contains a PyReverse exported UML class diagram.
Now that it works this method can be used to add diagrams to the software documentation.

.. graphviz:: 

    digraph "classes" {
    charset="utf-8"
    rankdir=BT
    "0" [label="{BalanceBotVrepEnvNoise|action_space : Box\lcamera\lgoal\lgoal_in_robot_frame : bool\lgoal_max : int\lgoal_mode_on : bool\lgoal_space : Box\ljoints_max_velocity : int\ll_angle\ll_wheel_delta : float\ll_wheel_old : float\llin_vel\llin_vel_old : recarray, list\lmax_torque : float\lmetadata : dict\lmin_torque : float\lobservation : list, recarray\lobservation_space : Box\loh_joint : list\loh_shape : list\lpitch_offset\lr_angle\lr_wheel_delta : float\lr_wheel_old : float\lsample_rate : int\lsteps : int\lverbose : bool\l|add_sensor_noise()\lcompute_action()\lcompute_reward()\lremap_observations()\lrender()\lreset()\lsample_goal()\lseed()\lstep()\l}", shape="record"];
    "1" [label="{Env|action_space : NoneType\lmetadata : dict\lobservation_space : NoneType\lreward_range : tuple\lspec : NoneType\lunwrapped\l|close()\lrender()\lreset()\lseed()\lstep()\l}", shape="record"];
    "2" [label="{VrepEnv|cID : int\lconnected : bool\lis_headless\lopM_get : int\lopM_set : int\lscene_loaded : bool\lscene_path : NoneType\lserver_addr\lserver_port\lsim_running : bool\lstr_simx_return : list\l|RAPI_rc()\ladd_statusbar_message()\lcall_childscript_function()\lclose()\lclose_scene()\lconnect()\ldisconnect()\lget_array_parameter()\lget_boolean_parameter()\lget_collision_handle()\lget_float_parameter()\lget_float_signal()\lget_integer_parameter()\lget_integer_signal()\lget_object_handle()\lget_string_signal()\lload_scene()\lobj_get_joint_angle()\lobj_get_joint_angle_continuous()\lobj_get_joint_force()\lobj_get_orientation()\lobj_get_orientation_continuous()\lobj_get_position()\lobj_get_velocity()\lobj_get_vision_image()\lobj_read_force_sensor()\lobj_set_force()\lobj_set_orientation()\lobj_set_position()\lobj_set_position_target()\lobj_set_velocity()\lread_collision()\lset_array_parameter()\lset_boolean_parameter()\lset_float_parameter()\lset_float_signal()\lset_integer_parameter()\lset_integer_signal()\lset_string_signal()\lstart_simulation()\lstep_simulation()\lstop_simulation()\l}", shape="record"];
    
    "0" -> "2" [arrowhead="empty", arrowtail="none"];
    "2" -> "1" [arrowhead="empty", arrowtail="none"];
    }

   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

