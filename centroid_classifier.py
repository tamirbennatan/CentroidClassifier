# -*- coding: utf-8 -*-

"""
The :mod:`centroid_classifier` implements a a generalization of the clasical
nearest centroid classification algorithm, available on Sklearn here:
http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html

This implementation makes two main generalizations of the implementation above. First, 
several additional distance metrics are implemented beyond those in Sklearn's module,
including a novel distance metric which takes into account the skew in the training
data along its principle components - proposed by Dr. Mark DeBonis in his paper 
"Using skew for classification".

The second generalization is that instead of predicting a class of a vector based on the
centroid it is nearest, the data is projected to a new vector space - where the components
of each projected vector are the distances to each of the centroids - and then arbitrary 
classification methods can be applied to this projected dataest. Using an LDA classifier
yields the classical classification method.
"""

# Author: Tamir Bennatan <timibennatan@gmail.com>


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array, check_consistent_length
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from utils.distance_factories import euclidean_dist_factory, manhattan_dist_factory,\
    mahalanobis_dist_factory, chebyshev_dist_factory
from utils.transform import class_subset, project_matrix, project_pca_2D
from utils.plotting import make_meshgrid, plot_contours

import pdb


distance_func_factories = {
    "euclidean" : euclidean_dist_factory,
    "manhattan" : manhattan_dist_factory, 
    "mahalanobis" : mahalanobis_dist_factory,
    "chebyshev" : chebyshev_dist_factory
    }


class CentroidClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, distance = "mahalanobis", classifier = LinearDiscriminantAnalysis()):
    	if distance is None:
    		distance = "mahalanobis"
    	# see if passed distance metric is a valid one
    	if distance not in distance_func_factories:
    		raise ValueError("Invalid distance measure `%s`. " % distance)
        self._distance = distance
        if classifier is None:
        	classifier = LinearDiscriminantAnalysis()
        self._classifier = classifier

    def set_params(self, **params):
        try:
            self.set_params(**params)
            return(self)
        except:
            self._classifier.set_params(**params)
            return(self)


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

    def plot_boundary(self, X, y,  title = None, alpha = .8):
        """
        Plot the decision boundary of a fitted model as a countour plot.
        Currently only supported for binary classification problems 

        If the data has more than 2 features, default behaviour is to project training vectors
        onto their 2 largest principle components, and plot a 2D countour plot. 
        """

        # verify that estimator was fitted successfully
        check_is_fitted(self, "_centroid_dist_factories")


        # Verify that response vector only has two classes
        if (len(self._classes) != 2):
            raise NotImplementedError("Decision boundary plotting only supported for binary \
                classification tasks. Fitted data has %d unique classes." % len(self._classes))
        
        # project the matrix `X` to vectors of distances from cluster centroids
        X_proj = project_matrix(X, self._centroid_dist_factories)

        # isolate the distances from first and second cluster centoids
        X1, X2 = X_proj[:,0], X_proj[:,1]

        # create a mesh to plot decision boundary onto
        xx, yy = make_meshgrid(X1, X2)

        # convert the labels to  integer factors
        y = pd.Categorical(y).codes

        f, ax = plt.subplots()

        plot_contours(ax, self._classifier, xx, yy,
                          cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X1, X2, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k', alpha = alpha)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('%s distance from first cluster centroid' %(self._distance.title()))
        ax.set_ylabel('%s distance from second cluster centroid' % (self._distance.title()))
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title("Decision boundary using %s distance." % (self._distance))

