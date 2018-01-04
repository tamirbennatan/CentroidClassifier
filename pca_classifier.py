# -*- coding: utf-8 -*-

"""

"""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array, check_consistent_length
from sklearn.externals import six
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from utils.disance_factories import euclidean_dist_factory, manhattan_dist_factory,\
	mahalanobis_dist_factory, chebyshev_dist_factory


distance_func_factories = {
	"euclidean" : euclidean_dist_factory,
	"manhattan" : manhattan_dist_factory, 
	"mahalanobis" : mahalanobis_dist_factory,
	"chebyshev" : chebyshev_dist_factory
	}


class PCAClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, classifier, distance = "mahalanobis"):
		self.classifier = OneVsRestClassifier(LinearDiscriminantAnalysis())
		self._distance = distance
		self._classes = None
		self._centroid_distances = None
		

	def fit(self, X, y):

		# convert X and y into numpy array-like of appropriate shape
		X = np.asanyarray(X)
		y = y.flatten()

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
		self._centroid_distances = []
		for c in self._classes:
			X_c = class_subset(X, y, c)
			self._centroid_distances.append(dist_factory(X_c))

		


		"""
		Project training data into `k` dimensions - where k is then number of classes.
		To do so, 


		"""








	def predict(self, X):
		pass



























