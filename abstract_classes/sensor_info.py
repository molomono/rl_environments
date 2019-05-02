import numpy as np

class SensorInfo: 
    # Decleration of the balancebot sensor types, names and noises
	#structure of the dict is:
	# Sensor_name
	# -- value names
	# -- sensor domain
	# -- -- lower bound
	# -- -- higher bound
	# -- sensor noise
	# -- -- mean
	# -- -- covariance (asuming uncorrilated sensor noise so it is a 1D list, not covar matrix)
    
    # TODO: Add datatype declerations, float, complex, integer

    sensor_dict = {
        'imu': {
            'labels': ['x_accel', 'y_accel', 'z_accel', 'x_gyro', 'y_gyro', 'z_gyro'], #Names of the sensors this is for developer readability
            'domain': {
                'init': [0, 0, 0, 0, 0, 0],  #Start position of the sensor
                'low': [0, 0, 0, 0, 0, 0],   #Lower range of sensors
                'high': [0, 0, 0, 0, 0, 0], #Upper range of sensors
            },
            'noise': {
                'mean': [0, 0, 0, 0, 0, 0],
                'covar': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            },
        },
        'wheel_vel': {
            'labels': ['right', 'left'],
            'domain': {
                'init': [0, 0],
                'low': [0, 0],
                'high': [0, 0],
            },
            'noise': {
                'mean': [0., 0.],
                'covar': [0.1, 0.1],
            },
        },
        'odom':{
            'labels': ['x', 'y', 'theta'],
            'domain': {
                'init': [0,0, 1+0j],
                'low': [0, 0, 0],
                'high': [0, 0, 0],
            },
            'noise': {  #noise on the odom can be used to impose drift in simulation
                'mean': [0., 0.],
                'covar': [0.0, 0.0],
            },
        },
    }

    def set_odom_init(self, odom):
        ''' Set the initialization value of the odom, used in conjunction with get_relative_pose() to calculate relative positioning
        '''
        x, y, theta = odom
        if np.iscomplexobj(theta):
            self.sensor_dict["odom"]["domain"]["init"] = [x, y, theta]
        else:
            self.sensor_dict["odom"]["domain"]["init"] = [x, y, complex(np.cos(theta), np.sin(theta))]

    def get_relative_pose(self, odom):
        ''' Converts the odometry information to relative positioning w.r.t. the last set odom 'init' point
        '''
        xy_relative = np.array(odom[0:2]) - np.array(self.sensor_dict["odom"]["domain"]["init"][0:2])
        if np.iscomplexobj(odom[2]):
            theta = self.sensor_dict["odom"]["domain"]["init"][2] * odom[2]
            return list(xy_relative) + [theta]
        else:
            # If the input is euler the math is still complex but the function returns euler
            theta = self.sensor_dict["odom"]["domain"]["init"][2] * complex(np.cos(odom[2]), np.sin(odom[2]) )
            return list(xy_relative) + [np.arctan2( theta.imag, theta.real )]
        
    def get_n_sensors(self):
        ''' Parse the sensor dictionary and returns the total number of sensors.
        '''
        n_sensors = 0
        for key in self.sensor_dict.keys():
            n_sensors += len(self.sensor_dict.get(key)["labels"] )
        return n_sensors

    def get_sensor_labels(self):
        ''' Parse the sensor dictionary and returns sensor labels
        '''
        labels = []
        for key in self.sensor_dict.keys():
            labels += [key, self.sensor_dict.get(key)["labels"]]
        return labels

    def get_observation_domain(self):
        ''' Parse the sensor dictionary and returns high, low values of the observation space
        '''
        high = []
        low = []
        for key in self.sensor_dict.keys():
            high += self.sensor_dict.get(key)["domain"]["low"]
            low += self.sensor_dict.get(key)["domain"]["high"]
        return high, low

    def get_sensor_noise(self):
        ''' Parse the sensor dictionary and returns the mean and covariance of the sensors.
        '''
        mean = []
        covar = []
        for key in self.sensor_dict.keys():
            mean = mean + self.sensor_dict.get(key)["noise"]["mean"]
            covar = covar + self.sensor_dict.get(key)["noise"]["covar"]
        return mean, covar



if __name__=="__main__":
    sensors = SensorInfo()
    
    print(sensors.get_n_sensors())
    
    print(sensors.get_sensor_labels())
    
    print(sensors.get_observation_domain())
    
    mean, covar = sensors.get_sensor_noise()
    print(np.random.normal( mean, covar))

    print('-------------------------------------')
    print(sensors.get_relative_pose([1,2,0+1j]))
    print(sensors.sensor_dict["odom"]["domain"]["init"][2])
    sensors.set_odom_init( [1, 2, np.pi/4] )
    print(sensors.sensor_dict["odom"]["domain"]["init"][2])
    print(sensors.get_relative_pose([1,2, np.pi*4])) 