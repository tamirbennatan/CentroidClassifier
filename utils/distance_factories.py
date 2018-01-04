"""

"""

import numpy as np
from scipy.spatial.distance import euclidean, cityblock, mahalanobis, chebyshev



def class_average(X):
	"""
	Compute average vector of a set of rows stacked (as rows) in a matrix, X. 
	This is used to find the centroid of each class in the training data.

	Parameters
	----------
	X : array-like, shape (n_samples, n_features)
	    Training vectors, where n_samples is the number of samples and
	    n_features is the number of features.

	Returns
	----------
	mean_vector: numpy.ndarray
		Row vector - Euclidean average of vectors in X. 
	"""
	# compute average row vector
	mean_vector = np.mean(X, axis = 0)
	# convert to float dtype
	mean_vector = mean_vector.astype(dtype = np.float64)
	return(mean_vector)



def euclidean_dist_factory(X):
	"""

	"""
	avg = class_average(X)
	def euclidean_distance(x):
		d = euclidean(x, v = avg)
		return(d)

	return(euclidean_distance)

def manhattan_dist_factory(X):
	"""

	"""
	avg = class_average(X)
	def cityblock_distance(x):
		d = cityblock(x, v = avg)
		return(d)

	return(cityblock_distance)

def mahalanobis_dist_factory(X):
	"""

	"""
	# compute the average vector location
	avg = class_average(X)
	# compute the variance-covariance matrix of the input matrix
	varcovar = np.cov(X, rowvar=False)
	# compute the inverse of the variance-covariance matrix
	inv_varcovar = np.linalg.pinv(varcovar)

	# function to compute Mahalanobis distance from vector `x` to class centroid 
	def mahalanobis_distance(x):
		d = mahalanobis (x, v = avg, VI = inv_varcovar)
		return(d)

	return(mahalanobis_distance)

def chebyshev_dist_factory(X):
	"""

	"""
	# compute the average vector location
	avg = class_average(X)

	# function to compute Chebyshev distance from vector `x` to class centroid 
	def chebyshev_distance(x):
		d = chebyshev (x, v = avg)
		return(d)

	return(chebyshev_distance)




