import numpy as np



def remap(value, input_min, input_max, output_min=0., output_max=1.):
    ''' Remaps the input value from input range to output range
	:param value: input value or numpy array
	:param input_min: minimum value of the input range
	:param input_max: maximum value of the input range
	:param output_min: minimum value of the output range
	:param output_max: maximum value of the output range
	:returns: input value or numpy array remapped to the output range
	'''
    # Convert the input to 0-1 range
    valueScaled = np.divide(value - input_min, (input_max - input_min))
    # Convert the 0-1 range into a value in the right range.
    return output_min + (np.multiply(valueScaled, (output_max - output_min)))

def sigmoid(x):
	''' Sigmoid function
    :param x: x value
    :returns: y = f(x)
	'''
	return 1.0 / (1.0 + np.exp(-x))

def gaussian(x,sigma=1.0):
	''' Gaussian distribution function
	:param x: Value on the x axis of the distribution
	:param sigma: sigma defines the distribution width
	:returns: y value, the probability of x occuring 
	'''
	return np.exp(-np.power(sigma*x,2.0))

def gaussian_2d(x,y , sigma_x=1.0, sigma_y=1.0):
	''' Two dimentional Gaussian distribution function
	:param x: Value on the x axis of the distribution
	:param y: Value on the y axis of the distribution
	:param sigma_x: distribution width in the x axis
	:param sigma_y: distribution width in the y axis
	:returns: Probability of f(x,y) 
	'''
	return 1/(np.e*np.pi*sigma_x*sigma_y) \
            * np.exp(-np.power(x,2.0) /(2*np.power(sigma_x,2.0))) \
            * np.exp(-np.power(y,2.0) /(2*np.power(sigma_y,2.0)))  

def trig(angle):
	''' Converts angle to a vector on the unit circle
	:param angle: Angle in radians
	:returns: x, y of the unit-vector
	'''
	return np.cos(angle), np.sin(angle)

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
	return np.matrix(np.dot(Rotate_Z_matrix,np.dot(Rotate_Y_matrix,np.dot(Rotate_X_matrix,Translate_matrix))))
