"""
Utilities for plotting.
Used in the static method `CentroidClassifier.plot_boundary`.
"""

import numpy as np
import matplotlib.pyplot as plt

def make_meshgrid(x, y, h=.02):
    """
    Create a mesh of points defined by two vectors, `x` and `y`. 
    Credit to Sklearn developers: http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = np.min(x) - 1, np.max(x) + 1
    y_min, y_max = np.min(y) - 1, np.max(y) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """
    Plot the decision boundaries for a classifier.
    Credit to Sklearn developers: http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out