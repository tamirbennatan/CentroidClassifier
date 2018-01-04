# -*- coding: utf-8 -*-

"""

"""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array, check_consistent_length
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from utils.distance_factories import euclidean_dist_factory, manhattan_dist_factory,\
	mahalanobis_dist_factory, chebyshev_dist_factory
from utils.transform import class_subset, project_matrix


distance_func_factories = {
	"euclidean" : euclidean_dist_factory,
	"manhattan" : manhattan_dist_factory, 
	"mahalanobis" : mahalanobis_dist_factory,
	"chebyshev" : chebyshev_dist_factory
	}


class PCAClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, classifier = OneVsRestClassifier(LinearDiscriminantAnalysis()), distance = "mahalanobis"):
		self._classifier = classifier
		self._distance = distance
		

	def fit(self, X, y):

		# convert X and y into numpy array-like of appropriate shape
		X, y = check_X_y(X, y)

		# store the unique response classes
		self._classes = np.unique(y)

		"""
		Get the distance function factory. 

		This factory returns a function that, after trained, returns 
		the distance from a vector the the centroid of the data the function was trained on.
		"""
		dist_factory = distance_func_factories[self._distance]

		"""
		For each class, store a function which computes the distance
		from a vector `x` to the centroid of the training data of that class. 
		"""
		self._centroid_dist_factories = []
		for c in self._classes:
			X_c = class_subset(X, y, c)
			self._centroid_dist_factories.append(dist_factory(X_c))

		"""
		Project training data into `k` dimensions - where k is then number of classes.

		To do so, generate functions that compute the distance from a vector to each of the
		cluster centroids. Then apply these functions to `X`, so that each training vector
		yields `k` points - corresponding to the distances to the `k` cluster centroids. 
		"""
		X_proj = project_matrix(X, self._centroid_dist_factories)

		"""
		Fit the inputed classifier to the projected dataset. 
		"""
		self._classifier.fit(X_proj, y)

		return self



	def predict(self, X):

		# verify that estimator was fitted successfully
		check_is_fitted(self, "_centroid_dist_factories")
		
		# check validity of input array shape
		X = check_array(X)

		# project the matrix `X` to vectors of distances from cluster centroids
		X_proj = project_matrix(X, self._centroid_dist_factories)

		# predict using trained classifier
		y_hat = self._classifier.predict(X_proj)

		return(y_hat)




