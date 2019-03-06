# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 07:49:41 2019

@author: Jongmin Sung

Parameter estimation of exponential distribution with t_max

"""

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import scipy 

# User parameters
n_dataset = 1000
n_sample = 10000
t_true = 1
t_max = range(1,11)
n_iter = 50



def main():
    
    # List of dataset that contains list of sample
    t_dataset = [None] * n_dataset
    
    # Array to save the result [row, col] = [dataset, t_max]
    result = np.zeros((n_dataset, len(t_max)))    
    
    # Generate array of exponential random variables
    for i in range(n_dataset):
        t_sample = np.random.exponential(t_true, n_sample)
        t_dataset[i] = t_sample        
        for j, t_cut in enumerate(t_max):
            t_sample_cut = t_sample[t_sample < t_cut]
            tau = t_sample_cut.mean() # Initial guess of tau
            for k in range(n_iter): # Iteratively find the original t_mean w/o cutoff
                tau = t_sample_cut.mean() + t_cut / (np.exp(t_cut/tau) - 1)        
            result[i][j] = tau # Corrected t_mean after n_iter
    
    
    # Plot_Convergece ---------------------------------------------------------
    t_cut = 2
    t_sample = t_dataset[0]
    t_sample_cut = t_sample[t_sample < t_cut]
    
    tau = np.zeros(n_iter+1)
    tau[0] = t_sample_cut.mean() # Initial guess of tau
    for k in range(n_iter): # Iteratively find the original t_mean w/o cutoff
        tau[k+1] = t_sample_cut.mean() + t_cut / (np.exp(t_cut/tau[k]) - 1)  
    

    t = np.linspace(0, t_sample_cut.max(), 100)    

    fig = plt.figure('Convergence', clear=True)    
        
    sp = fig.add_subplot(131)
    sp.plot(tau, 'k')
    sp.axhline(y=t_true, color='k', linestyle='dotted', lw=1)   
    sp.set_xlabel('Iteration')
    sp.set_ylabel('Estimate of t_mean')

    sp = fig.add_subplot(132)
    sp.hist(t_sample, 50, color='k', histtype='step', density='True', lw=1)
    sp.plot(t, np.exp(-t/t_true), 'k')
    sp.plot(t, np.exp(-t/t_sample_cut.mean()), 'b')
    sp.plot(t, np.exp(-t/tau[-1]), 'r')
    sp.set_xlim([0, t.max()])
    sp.set_ylim([0, 1])
        
    sp = fig.add_subplot(133)
    sp.hist(t_sample, 50, color='k', histtype='step', density='True', lw=1)     
    sp.plot(t, np.exp(-t/t_true), 'k')      
    sp.plot(t, np.exp(-t/t_sample_cut.mean()), 'b')        
    sp.plot(t, np.exp(-t/tau[-1]), 'r')
    sp.set_yscale('log')
    sp.set_xlim([0, t.max()])           
    sp.set_ylim([np.exp(-t_cut/t_sample_cut.mean()), 1])
 
    # Plot_result -------------------------------------------------------------

    t_est_mean = result.mean(axis=0) / t_true
    t_est_std = result.std(axis=0) / t_true
    t_max_true = np.array([i/t_true for i in t_max])
    
    fig = plt.figure('Estimate', clear=True)    
        
    sp = fig.add_subplot(111)
    sp.errorbar(t_max_true, t_est_mean, yerr = t_est_std)
    sp.axhline(y=1, color='k', linestyle='dotted', lw=1)   
    sp.set_xlabel('t_max / t_true')
    sp.set_ylabel('t_estimate / t_true')    
    
    

if __name__ == "__main__":
    main()



