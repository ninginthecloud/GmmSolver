
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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.style.use('ggplot')
import seaborn as sns
sns.set_style("darkgrid")
from scipy.stats import norm
from sklearn import mixture
import time
from gmm_poly import *
from gmm_test import *
get_ipython().magic(u'matplotlib inline')


# In[11]:

class Experiment(object):
    def __init__(self, param,seed = 1):
        self.param = param
        #self.n_samples = n_samples
        self.seed = seed
        #self.Test = Test(param = self.param,n_samples=self.n_samples, seed = self.seed)
    
    def distrate(self, eps, n_samples, n_reps = 50):
        d1, d2 = [], []
        for i in np.arange(n_reps):
            temptest = Test(param = self.param,n_samples= n_samples, seed = np.random.choice(i+self.seed, 1))
            res1, res2 = temptest.unitest(eps,isplot = False)
            d1 += [res1[1]]
            d2 += [res2[1]]
        return [np.mean(np.array(d1)/np.array(d2)), np.nanmean(d1), np.nanmean(d2)]
    
    def distrate_test(self, eps, samplelist, filename = '../man/plots/rate.pdf',isplot = True):
        return
    
    def timerate(self, eps, n_samples, n_reps=50):
        t1, t2 = [], []
        d1, d2 = [], []
        for i in np.arange(n_reps):
            temptest = Test(param = self.param,n_samples= n_samples, seed = np.random.choice(i+1, 1))
            res1, res2 = temptest.unitest(eps,isplot = False)
            t1 += [res1[0]]
            t2 += [res2[0]]
            d1 += [res1[1]]
            d2 += [res2[1]]
        return [np.mean(np.array(t1)/np.array(t2)), np.mean(t1), np.mean(t2), np.mean(d1), np.nanmean(d2)]
        #return res2
        
    def timerate_test(self, eps, samplelist, filename = '../man/plots/rate.pdf',isplot = True):
        ratelist, t_sklearn, t_moment = [], [], []
        p_sklearn, p_moment = [], []
        for s in samplelist:
            print "running on rep = {}".format(s)
            temp = self.timerate(eps, s) 
            ratelist += [temp[0]]
            t_sklearn += [temp[1]]
            t_moment += [temp[2]]
            p_sklearn += [temp[3]]
            p_moment += [temp[4]]
        if isplot:
            self.time_plot(samplelist, ratelist, t_sklearn, t_moment, p_sklearn, p_moment )
        return [p_sklearn, p_moment]
    
    def dist_plot(self, sample, dist_sklearn, dist_moment, param_sklearn, param_moment):
        plt.figure()
        sns.set_style("darkgrid")
        plt.plot(sample, time_sklearn, marker='o', linestyle='--', color='r', label='sklearn')
        plt.plot(sample, time_moment, marker='o', linestyle='--', color='b', label='moment')
        plt.xlabel('SampleSize')
        plt.ylabel('Parameter distance')
        plt.legend(loc = 'upper left')
        plt.show()
        
    def time_plot(self, sample, rate, time_sklearn, time_moment, param_sklearn, param_moment):
        plt.figure()
        sns.set_style("darkgrid")
        plt.plot(sample, rate, marker='o', linestyle='-', color='r', label='Rate(sklearn/moment)')
        plt.xlabel('SampleSize')
        plt.ylabel('Rate(sklearn/moment)')
        plt.legend()
        plt.savefig('../man/plots/rate.pdf')
        
        plt.figure()
        sns.set_style("darkgrid")
        plt.plot(sample, time_sklearn, marker='o', linestyle='--', color='r', label='sklearn')
        plt.plot(sample, time_moment, marker='o', linestyle='--', color='b', label='moment')
        plt.xlabel('SampleSize')
        plt.ylabel('Time')
        plt.legend(loc = 'upper left')
        plt.savefig('../man/plots/time.pdf')
        
        plt.figure()
        sns.set_style("darkgrid")
        plt.loglog(sample, param_sklearn, marker='o', linestyle='--', color='r', label='sklearn')
        plt.loglog(sample, param_moment, marker='o', linestyle='--', color='b', label='moment')
        plt.xlabel('SampleSize')
        plt.ylabel('parameter distance')
        plt.legend(loc = 'upper right')
        plt.savefig('../man/plots/parameter_distance.pdf')


# In[12]:

if __name__ == "__main__":
    experiment_turn1 = Experiment(param = [0.5, 0.5, -2, 2, 1,1], seed = 50)
    #dtest = experiment_turn1.timerate(.5, 1000)
    candlist = [50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    #candlist = [100, 1000, 10000, 100000, 1000000]
    plotlist = experiment_turn1.timerate_test(0.1, candlist, isplot = True)

