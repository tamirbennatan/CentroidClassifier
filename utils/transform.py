"""
Data tranformation utilities.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def class_subset(X, y, c):
	"""
	Subset data (X) based on which response falls in a certain class

	Parameters
	----------
	X : array-like, shape (n_samples, n_features)
	    Design matrix, where n_samples is the number of samples and
	    n_features is the number of features.

	y : array-like. shape (n_samples,) or (1, n_samples)
		Response vector. Value at index `i` is label to training vector 
		in row `i` of X

	class : scalar
		A response class. 


	Returns
	----------
	subset: array-like
		subset = `class_subset(X, y, c)` returns the rows in `X` 
		corresponding the indecies in `y` where `y[i] == class`.
	"""
	subset = X[np.where(y == c),][0]	
	return(subset)

def project_matrix(X, funcs):
	"""
	Given a design matrix `X`, and a list of functions `funcs`, 
	apply these functions in sequence to the columns of `X` to yield a list of np.arrays, 
	then concatenate the result. 

	This is used primarily to project a series of training vectors to vectors which are the
	distances from each vector to each response class centroid.

	Parameters
	----------
	X : array-like, shape (n_samples, n_features)
	    Design matrix, where n_samples is the number of samples and
	    n_features is the number of features.

	funcs : list of functions
		Functions to apply to the columns of `X`. 
		Functions are expected to have the behaviour of `f(l) -> l'), where
		l is a list or numpy array


	Returns
	----------
	X_proj : array-like, shape (n_samples, k)
		Each vector corresponds to a vector of dimension `k`, 
		where `k` is the output dimension of each of the functions in `funcs`. 
	"""
	# generate functions to compute distance from a vector to each cluster centroid
	# then apply these functions to X to get an array of length `n_samples`, 
	# which correspond to the distances from each point to each cluster centroid.
	dists = [np.apply_along_axis(f, 1, X) for f in funcs]
	# Concatenate the distances from each point to each cluster centroid to get a 
	# projected dataset, X_proj
	X_proj = np.stack(dists, axis = 1)

	return(X_proj)


def pca_basis(X, n_components = None):
	"""
	Retrieve the PCA basis of a matrix, X.

	Parameters
	----------
	X : array-like, shape (n_samples, n_features)
	    Matrix - numeric data type. 

	Returns
	----------
	pca_basis : Array-Like, shape (n_features, n_features)
		Trained PCA decomposition.
	"""
	# unless stated otherwise, retrieve the basis of the smame dimension of X. 
	if n_components is None:
		n_components = X.shape[1]
	pca_basis = PCA(n_components=2).fit_transform(X)
	return(pca_basis)















