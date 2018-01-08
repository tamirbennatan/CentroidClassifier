# -*- coding: utf-8 -*-
"""
@author: Rocco Pascale
@email: rpascale.student@manhattan.edu

Here we have all the functions needed to compute the skew distance for
the centroid classifier.  This is all based on the work done by Dr. DeBonis
in his paper "Using Skew for Classification."

"""

from __future__ import division
# True division.

import numpy as np
import scipy.stats as sci
# Packages for arrays and stats.

from sklearn.neighbors import KernelDensity
# ML packages.

def mode_Finder(x):
    """
    Function to find the mode of a Univariate dataset.
    This is found via Kernal Density Estimation.
    In the case of a multimodal estimate, the maximum mode is taken.
    
    Parameters
	 ----------
	 x : array-like, shape (n_samples,)
	     Array of data of which to find the mode.

    Returns
	 ----------
	 mode_x : scalar
             The maximum mode of the distribution of the dataset.
         
    """
    
    silver_BW = np.std(x)*(4/3/np.shape(x)[0])**(1/5)
    # Silverman's estimate for bandwidth.
    
    kde_x = KernelDensity(bandwidth = silver_BW, 
                          rtol = 1E-4).fit(x.reshape(-1,1))
    # Fitted kde for each feature.
    
    support_x = np.linspace(np.min(x), np.max(x), num = 1000)
    # Support to fit density on, based on empirical data.
    
    x_Fit = np.exp(kde_x.score_samples(support_x.reshape(-1,1)))
    # Fit to empirical support.

    mode_x = support_x[x_Fit == np.max(x_Fit)][-1]
    # Finding the mode of our feature.
    # In the case the distribution is multimodal, take the largest mode.
    
    # plt.plot(support_x, x_Fit)
    # Plotting if curious.
    
    return(mode_x)

def skew_Normal_Fit(x):
    """
    Function to find the MLE parameters for a skew normal fit to the data.
    This uses the closed forms for these parameters from Dr. DeBonis's paper.
    
    Parameters
	 ----------
	 x : array-like, shape (n_samples,)
	     Array of data to fit.

    Returns
	 ----------
	 parameters : dictionary of floats
                 k is the MLE of this parameter.
                 sigma is the MLE of the St. Dev.
         
    """
    
    N = np.shape(x)[0]
    # Number of observations.
    
    pos_Obs = x[x >= 0]
    neg_Obs = x[x < 0]
    # Here we seperate our observations according to the pairty.
    
    a = np.dot(neg_Obs, neg_Obs)
    # Finding the sum of squares of our negative observations.
    b = np.dot(pos_Obs, pos_Obs)
    # Finding the sum of squares of our positive observations.
    
    k = (b/a)**(1/6)
    # MLE for k.
    sigma = np.sqrt((a*k**2 + b/k**2) / N)
    # MLE for sigma.
    
    parameters = {
    "k" : k,
    "sigma" : sigma, 
    }
    # Output Dictionary.
    
    return(parameters)

def skew_Laplace_Fit(x):
    """
    Function to find the MLE parameters for a skew laplace fit to the data.
    This uses the closed forms for these parameters from Dr. DeBonis's paper.
    
    Parameters
	 ----------
	 x : array-like, shape (n_samples,)
	     Array of data to fit.

    Returns
	 ----------
	 parameters : dictionary of floats
                 k is the MLE of this parameter.
                 sigma is the MLE of the St. Dev.
         
    """    
    
    N = np.shape(x)[0]
    # Number of observations.
    
    pos_Obs = x[x >= 0]
    neg_Obs = -1*x[x < 0]
    # Here we seperate our observations according to the pairty.
    # We also take the abs of our negative observations.
    
    a = np.sum(neg_Obs)
    # Finding the sum of the abs of our negative observations.
    b = np.sum(pos_Obs)
    # Finding the sum of the positive observations.
    
    k = (b/a)**(1/4)
    # MLE for k.
    sigma = ((a*k + b/k) / N) * np.sqrt(2)
    # MLE for sigma.
    
    parameters = {
    "k" : k,
    "sigma" : sigma, 
    }
    # Output Dictionary.
    
    return(parameters)

def skew_Gen_Fit(x):
    """
    Function to find the MLE parameters for a skew generalized Cauchy fit to the data.
    Dr. DeBonis has found that the MLE of k is the intersection of two curves.
    Note, it is proven that these curves are guarenteed a unique intersection.
    Since we numerically find this, we simply find the point (over a span of k)
    which has the minimum absolute difference between the two curves.  For a,
    we have two closed form estimates, and simply average them to reduce variance.
    
    Dr. DeBonis has also found closed forms for the 1st, 2nd, and 3rd central 
    moments of this distribution.  Once we have these parameters, we can simply
    find these via computation.
    
    Parameters
	 ----------
	 x : array-like, shape (n_samples,)
	     Array of data to fit.

    Returns
	 ----------
	 parameters : dictionary of floats
                 k is the MLE of this parameter.
                 a is the MLE of the parameter.
         
    """ 
    # We look to find the intersection
    # (or closest point) of these two functions. 
    
    N = np.shape(x)[0]
    # Number of observations.
    
    k = np.linspace(0.001, 100, 1000)
    # Values to evaluate y_3 and y_4 over.  
    
    pos_Obs = x[x >= 0]
    neg_Obs = x[x < 0]
    # Here we seperate our observations according to the pairty.
    
    kxi = np.outer(neg_Obs, k)
    k_xi = np.outer(pos_Obs, 1/k)
    # Here we use an outer product to find each observation x, 
    # multiplied by each value of k, and 1/k respectively.
    
    y_3 = ((N*(k**2 - 1) / (k**2 + 1)) / 
           (np.sum(kxi / (1 - kxi), axis = 0) + 
            np.sum(k_xi / (1 + k_xi), axis = 0)))
    # Function for first curve.
    
    y_4 = (N / (np.sum(np.log(1 - kxi), axis = 0) + 
               (np.sum(np.log(1 + k_xi), axis = 0)))) + 1
    # Function for second curve.
    
    k_index = np.min(np.abs(y_3 - y_4)) == np.abs(y_3 - y_4)
    # Index for the intersection, or closest point.
    
    k_MLE = k[k_index]
    # Point of intersection, the MLE for k.
    
    a = np.mean((y_3[k_index], y_4[k_index]))
    # Mean of both Closed form MLEs for a.
    
    parameters = {
    "k" : k_MLE,
    "a" : a, 
    }
    # Output Dictionary.
    
    return(parameters)

def sigma_Finder(x):
    """
    Function to find the one-sided Std. Deviations of the fit data.
    This function utilizes skew_Normal_Fit, skew_Laplace_Fit, and
    skew_Gen_Fit to first fit the data.  We then use the closed forms of the
    one-sided Std. Deviations.  We use the moments discuseed in the doc. of
    skew_Gen_Fit here to find the Std. Deviation.
    
    Parameters
	 ----------
	 x : array-like, shape (n_samples,)
	     Array of data of which to find the one sided sigmas.

    Returns
	 ----------
	 sig_Dict : dictionary of floats
               pos_StD is the right side St. Dev.
               neg_StD is the left side St. Dev.
         
    """ 
    
    abs_Skew = np.abs(sci.skew(x))
    # Finding the absolute value of the skew of the input feature.
    # This is to ascertain which skew distribution we will use to fit the data.
    
    sig_Array = np.array([np.NaN, np.NaN])
    # Intializing the sig_Array for our variances.
    
    if abs_Skew <= 1:
        param_Dict = skew_Normal_Fit(x)
        # If the abs skew is in [0,1], we fit the data with a skew normal dist.
        
        sig_Array[0] = param_Dict['sigma'] * param_Dict['k']
        sig_Array[1] = param_Dict['sigma'] / param_Dict['k']
        # Closed forms for the one sided sigmas.
        
    elif abs_Skew <= 2:
        param_Dict = skew_Laplace_Fit(x)
        # If the abs skew is in (1,2], we fit the data with a skew laplace dist.
    
        sig_Array[0] = param_Dict['sigma'] * param_Dict['k']
        sig_Array[1] = param_Dict['sigma'] / param_Dict['k']
        # Closed forms for the one sided sigmas.
        
    else:
        param_Dict = skew_Gen_Fit(x)
        # If the abs skew is > 2, we fit the data with a skew gen Cauchy dist.
        
        k = param_Dict['k']
        a = param_Dict['a']
        # MLE for parameters from above fit.
        
        if a > 4:
        # We must check for this condition, as the Second Central Moment
        # does not exist otherwise.  As an adhoc solution, we use the next
        # fit which can handle skew: skew laplace.
        
            EX = (k**2 - 1) / (k * (a - 2))
            # First Central Moment.
            EX2 = (2*(k**4 - k**2 + 1)) / (k**2 * (a - 2) * (a - 3))
            # Second Central Moment.
            
            sigma = np.sqrt(EX2 - EX**2)
            # St.Dev via moments.
            
            sig_Array[0] = sigma  * param_Dict['k']
            sig_Array[1] = sigma / param_Dict['k']
            # Closed forms for the one sided sigmas.
            
        else:
            param_Dict = skew_Laplace_Fit(x)
            # If the abs skew is in (1,2], we fit the data with a skew laplace dist.
    
            sig_Array[0] = param_Dict['sigma'] * param_Dict['k']
            sig_Array[1] = param_Dict['sigma'] / param_Dict['k']
            # Closed forms for the one sided sigmas.
            
    return(sig_Array)

def skew_Dist(x, mode, sigmas):
    """
    Function to find the skew distance of a point to the centroid of
    a Data set.  This dataset Q_i, has been transformed by the operations
    in the parent function.  Q_i is described implictly by mode and sigmas.
    # This distance metric was proposed by Dr. DeBonis as a generalization of
    # Mahalanobis distance, to include skew in its calculation.
    
    Parameters
	 ----------
	 x : array-like, shape (n_samples,)
	     Data sample of which to find the skew distance from Q_i.
        Q_i is described by its mode and one-sided sigmas.
         
	 mode : array-like, shape (n_samples,)
           Mode of Q_i (for each feature).
         
	 sigmas : array-like, shape (n_samples,)
             One-sided St.Devs of Q_i (for each feature).
           
    Returns
	 ----------
	 sig_Dict : dictionary of floats
               pos_StD is the right side St. Dev.
               neg_StD is the left side St. Dev.
         
    """ 
    common_Term = (x - mode)
    # Common term for both expressions in the distance formula.
    
    pos_Dims = common_Term >= 0
    neg_Dims = common_Term < 0
    # Here we seperate the shifted dims of our observations
    # according to the pairty.
    
    sk_Dist = ( np.sum((common_Term[neg_Dims] / sigmas[1][neg_Dims])**2) + 
                np.sum((common_Term[pos_Dims] / sigmas[0][pos_Dims])**2 ))
    # Skew Distance formula.
    
    return(sk_Dist)
