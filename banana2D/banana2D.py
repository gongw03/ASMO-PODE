from __future__ import division, print_function, absolute_import
import numpy as np
from numba import jit

@jit
def logpdf(x):
    ''' banana 2D test function
        X: input vector, dim = 30, upper and lower bound limited to [-6, 6]
        return -2logpdf of high dimensional Gaussian
    '''
    b = 0.1
    mu = np.array([0.0, 0.0])
    # sigma = np.array([[100.0, 0.0],[0.0, 1.0]])
    # invsigma = np.linalg.inv(sigma)
    invsigma = np.array([[0.01, 0.0],[0.0, 1.0]])
    theta = np.array([x[0], x[1] + b*x[0]**2 - 100*b])
    
    # return the -2log(likelihood)
    # the const factor also ommited
    xmu = theta - mu
    logpdf = np.sum(xmu * invsigma * xmu.transpose())
    return logpdf

@jit
def evaluate(values):
    if len(values.shape) == 1: values = values.reshape((1,values.shape[0]))
    Y = np.empty([values.shape[0]])
    for i, x in enumerate(values):
        Y[i] = logpdf(x)
    return Y
