"""


"""

import numpy as np
import pandas as pd


def class_subset(X, y, c):
	"""
	Subset data (X) based on which response falls in a certain class

	Parameters
	----------
	X : array-like, shape (n_samples, n_features)
	    Design matrix, where n_samples is the number of samples and
	    n_features is the number of features.

	y : array-like. shape (n_samples,)
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
	subset = X[y == c,]	
	return(subset)

