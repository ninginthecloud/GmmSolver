
# coding: utf-8

# In[1]:

#! /usr/bin/env python
'''
    code for gaussian mixture solver
    based on Kalai STOC'10 paper
    Efficiently learning mixtures of two gaussians
'''
from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn import mixture
import time
from gmm_poly import *
get_ipython().magic(u'matplotlib inline')
class Test(object):
    def __init__(self, param, n_samples, seed = 1):
            self.w1 = param[0]
            self.w2 = param[1]
            self.mu1 = param[2]
            self.mu2 = param[3]
            self.sigma1 = param[4]
            self.sigma2 = param[5]
            self.n_samples = n_samples
            self.sample2d, self.sample1d  = self.sample()
            
    def sample(self, seed = 0):
        gmix = mixture.GaussianMixture(n_components=2, covariance_type='full')
        gmix.fit(np.random.rand(2,1))  # Now it thinks it is trained
        gmix.weights_ = np.array([self.w1, self.w2]) # mixture weights (n_components,) 
        gmix.means_ = np.array([[self.mu1], [self.mu2]])         # mixture means (n_components, 2) 
        gmix.covariances_ = np.array([[[self.sigma1**2]], [[self.sigma2**2]]]) # mixture cov (n_components, 2, 2)
        sample, _ = gmix.sample(self.n_samples)
        return (sample, sample.flatten())

    def unitest(self, eps, isplot = True):
        # fit a Gaussian Mixture Model with two components
        start = time.time()
        clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
        clf.fit(self.sample2d)
        end = time.time()
        t1 = (end - start)

        start = time.time()
        test = TwoGaussian(self.sample1d)
        #res = test.recoverFromMoments(eps)
        res = test.recover1DMixture(eps)
        end = time.time()
        t2 = (end - start)
        print "Skelearn GMM cost time is: {} sec. \n".format(t1)
        print "Moment method GMM cost time is: {} sec. \n".format(t2)
        param1 = self.parseGMM(clf)
        param2 = self.parseTwoGmm(res)
        if isplot:
            self.plot(param1.flatten(), param2.flatten())
        return (param1, param2)
    
    def parseGMM(self, model):
        return np.round([model.weights_.flatten(), model.means_.flatten(), np.sqrt(model.covariances_.flatten())],4)
    
    def parseTwoGmm(self, model):
        model = np.array(model)
        return np.round(model.real,4)
    def generateMixture(self, bins, param):
        assert len(param) == 6
        return param[0]*norm.pdf(bins, loc=param[2], scale=param[4])        +param[1]*norm.pdf(bins, loc=param[3], scale=param[5])
        
    def plot(self, param1, param2):
        n, bins, patches =plt.hist(self.sample1d, 50, normed=1, facecolor='green', alpha=0.75)
        y1 = self.generateMixture(bins, param1)
        plot1 = plt.plot(bins, y1, 'r--', linewidth=1, color = 'red')
        y2 = self.generateMixture(bins, param2)
        plot2 = plt.plot(bins, y2, 'r--', linewidth=1, color = 'blue')

