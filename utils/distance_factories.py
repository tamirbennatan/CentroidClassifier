"""
To train the centroid classifier in `centroid_classifier` module, need to learn
functions that compute the distance from an arbitrary vector to each cluster centroid.

Here are function factories, which - given a subset of data all form one class - 
return a function that computes the distance from an arbitrary vector to 
the centroid of that class.
"""

from __future__ import division
# True Division.

import numpy as np
from scipy.spatial.distance import euclidean, cityblock, mahalanobis, chebyshev
from sklearn.decomposition import PCA

from utils.Skew_Funcs import mode_Finder, sigma_Finder, skew_Dist
# Functions needed to compute the skew distance.



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
	return(mean_vector)

def euclidean_dist_factory(X):
	"""
	Create a function which computes the Euclidean distance to the
	centroid of `X`. 

	Parameters
	----------
	X : array-like, shape (n_samples, n_features)
	    Training vectors - assumed to all be part of one class.

	Returns
	----------
	euclidean_distance : function
		takes in a vector of shape (n_features,), and returns the 
		Euclidean distance to the centroid of `X`.
	"""
	avg = class_average(X)
	def euclidean_distance(x):
		d = euclidean(x, v = avg)
		return(d)

	return(euclidean_distance)

def manhattan_dist_factory(X):
	"""
	Create a function which computes the Manhattan distance to the
	centroid of `X`. 

	Parameters
	----------
	X : array-like, shape (n_samples, n_features)
	    Training vectors - assumed to all be part of one class.

	Returns
	----------
	cityblock_distance : function
		takes in a vector of shape (n_features,), and returns the 
		Manhattan distance to the centroid of `X`.
	"""
	avg = class_average(X)
	def cityblock_distance(x):
		d = cityblock(x, v = avg)
		return(d)

	return(cityblock_distance)

def mahalanobis_dist_factory(X):
	"""
	Create a function which computes the Mahalanobis distance to the
	centroid of `X`. 

	Parameters
	----------
	X : array-like, shape (n_samples, n_features)
	    Training vectors - assumed to all be part of one class.

	Returns
	----------
	mahalanobis_distance : function
		takes in a vector of shape (n_features,), and returns the 
		Mahalanobis distance to the centroid of `X`.
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
	Create a function which computes the Chebyshev distance to the
	centroid of `X`. 

	Parameters
	----------
	X : array-like, shape (n_samples, n_features)
	    Training vectors - assumed to all be part of one class.

	Returns
	----------
	chebyshev_distance : function
		takes in a vector of shape (n_features,), and returns the 
		Chebyshev distance to the centroid of `X`.
	"""
	# compute the average vector location
	avg = class_average(X)

	# function to compute Chebyshev distance from vector `x` to class centroid 
	def chebyshev_distance(x):
		d = chebyshev (x, v = avg)
		return(d)

	return(chebyshev_distance)


def skew_distance_factory(X):
    """
    @author: Rocco Pascale 
    @email: rpascale.student@manhattan.edu
    This is all based on the work done by Dr. DeBonis
    in his paper "Using Skew for Classification".
    
    Create a function which computes the Skew distance to the
    centroid of `X`. 
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    Training vectors - assumed to all be part of one class.
    
    Returns
    ----------
    skew_distance : function
    takes in a vector of shape (n_features,), and returns the 
    Skew distance to the centroid of `X`.
    """   
    
##### 1: Center Around Mean ###################################################
    
    C_i = X - class_average(X)
    # Centering the class around its mean.
    # Using the class_average function to do so.
##### 1: Center Around Mean ###################################################
    
##### 2: PCA Basis Change #####################################################
    
    pca_Basis = PCA().fit(C_i).components_
    P_i = np.matmul(C_i, pca_Basis)
    # Performing the PCA Basis change.
    # Utilizing the PCA object from sklearn.
##### 2: PCA Basis Change #####################################################
    
##### 3: Center Around Mode ###################################################
    
    feature_Modes = np.apply_along_axis(mode_Finder, 0, P_i)
    Q_i = P_i - feature_Modes
    # Finding the mode for each feature of Q_i and centering.
##### 3: Center Around Mode ###################################################
    
##### 4: Skew Distribution Fit and 1-sd Sigmas ################################
    
    feature_Sigmas = np.apply_along_axis(sigma_Finder, 0, Q_i)
    # Fitting each feature of Q_i according to the its skew.
    # Then using that fit to find the one-sided Std. Deviations.
    # First array is the Positive Std. Dev.
    # Second Array is the Negative Std. Dev.
##### 4: Skew Distribution Fit and 1-sd Sigmas ################################
    
##### 5: Skew Distance ########################################################
        
    def skew_distance(x):
        x_BC = pca_Basis.dot(x)
        # Changing x to the proper basis.
        
        d = skew_Dist(x_BC, mode = feature_Modes, sigmas = feature_Sigmas)
        # Skew distance function on a particular observation.
        return(d)
##### 5: Skew Distance ########################################################
    
    return(skew_distance)










