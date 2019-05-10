# Decleration of the balancebot sensor types, names and noises
# structure of the dict is:
# Sensor_name
# -- value names
# -- sensor domain
# -- -- lower bound
# -- -- higher bound
# -- sensor noise
# -- -- mean
# -- -- covariance (asuming uncorrilated sensor noise so it is a 1D list, not covar matrix)
    
# TODO: Add datatype declerations, float, complex, integer

loomo_sensor_dict = {
    'imu': {
        'labels': ['x_accel', 'y_accel', 'z_accel', 'x_gyro', 'y_gyro', 'z_gyro'], #Names of the sensors this is for developer readability
        'domain': {
            'init': [0., 0., 0., 0., 0., 0.],  #Start position of the sensor
            'low': [0., 0., 0., 0., 0., 0.],   #Lower range of sensors
            'high': [0., 0., 0., 0., 0., 0.], #Upper range of sensors
        },
        'noise': {
            'mean': [0., 0., 0., 0., 0., 0.],
            'covar': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        },
    },
    'wheel_vel': {
        'labels': ['right', 'left'],
        'domain': {
            'init': [0., 0.],
            'low': [0., 0.],
            'high': [0., 0.],
        },
        'noise': {
            'mean': [0., 0.],
            'covar': [0.01, 0.01],
        },
    },
    'odom_2d':{
        'labels': ['x', 'y', 'theta'],
        'domain': {
            'init': [0., 0., 1.+0.j],
            'low': [0., 0., 0.],
            'high': [0., 0., 0.],
        },
        'noise': {  #noise on the odom can be used to impose drift in simulation
            'mean': [0., 0.],
            'covar': [0., 0.],
        },
    },
}

import pickle

with open('loomo.pickle', 'wb') as handle:
    pickle.dump(loomo_sensor_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('loomo.pickle', 'rb') as handle:
    pickle_load = pickle.load(handle)

print(loomo_sensor_dict == pickle_load)