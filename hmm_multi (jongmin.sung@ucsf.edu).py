# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 19:56:50 2019 @author: Jongmin Sung

The code was modified from the following. 
https://hmmlearn.readthedocs.io/en/stable/index.html

Model: 
    Number of states = 2 [unbound, bound]
    Transition probability = [[tp_uu, tp_ub], [tp_bu, tp_bb]]
    Emission probability = [ep_u, ep_b]

"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy 
from hmmlearn import hmm
import random
import time


# User parameters
n_sample = 100
n_frame = 200
SNR = 20
time_bound = 10
time_unbound = 20

print(time_bound)
print(time_unbound)

# Start probability 
startprob = np.array([0.5, 0.5])

# The transition probability matrix
tp_ub = 1/time_unbound
tp_uu = 1 - tp_ub
tp_bu = 1/time_bound
tp_bb = 1 - tp_bu
transmat = np.array([[tp_uu, tp_ub],
                     [tp_bu, tp_bb]])

# The means of each state
means = np.array([[0.0], 
                  [1.0]])

# The covariance of each component
covars = np.tile(np.identity(1), (2, 1, 1)) / SNR**2

def icdf(data, time):
    data = np.array(data)
    cdf = np.zeros(len(time))
    for i in time:
        cdf[i] = sum(data <= time[i])
    icdf = 1 - cdf/max(cdf)
    return icdf

def reject_outliers(data, m = 5.):
    data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def Gaussian(m, s, x):
    return np.exp(-(x-m)**2/(2*s**2))/(2*np.pi*s**2)**0.5

def LL_G(param, x):      
    [m, s] = param
    return np.sum(np.log(Gaussian(m, s, x)))  

def MLE_G(x): 
    m = np.mean(x)
    s = np.std(x)
    fun = lambda *args: -LL_G(*args)
    p0 = [m, s]
    result = scipy.optimize.minimize(fun, p0, args=(x)) 
    return result

def Exp3(m, a, b, x):
    return a*np.exp(-x/m) + b 

def Exp1(m, x):
    return np.exp(-x/m) 

def Exp_pdf(m, x):
    return np.exp(-x/m)/m 

def LL_E(param, x):      
    [m] = param
    return np.sum(np.log(Exp_pdf(m, x)))  

def MLE_E(x):
    m = np.mean(x)
    fun = lambda *args: -LL_E(*args)
    p0 = [m]
    result = scipy.optimize.minimize(fun, p0, args=(x)) 
    return result
    

class Data:
    def __init__(self):
        pass


    def generate(self):       
        # Build an HMM instance and set parameters
        model = hmm.GaussianHMM(n_components=2, covariance_type="full")
        
        # Set the parameters to generate samples
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.means_ = means
        model.covars_ = covars        
        
        # Generate samples
        X, Z_true = model.sample(n_frame)
        return X, Z_true
        
    
    def predict(self, X):
        # Set a new model for traidning
        remodel = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)        
        
        # Set initial parameters for training
        remodel.startprob_ = np.array([0.5, 0.5])
        remodel.transmat_ = np.array([[0.5, 0.5], 
                                      [0.5, 0.5]])
        remodel.means_ = np.array([0, 1])
        remodel.covars_ = np.tile(np.identity(1), (2, 1, 1)) / SNR**2       
             
        # Estimate model parameters (training)
        remodel.fit(X)  
        
        # Find most likely state sequence corresponding to X
        Z_predict = remodel.predict(X)
        
        # Reorder state number such that X[Z=0] < X[Z=1] 
        if X[Z_predict == 0].mean() > X[Z_predict == 1].mean():
            Z_predict = 1 - Z_predict
            remodel.transmat_ = np.array([[remodel.transmat_[1][1], remodel.transmat_[1][0]],
                                          [remodel.transmat_[0][1], remodel.transmat_[0][0]]])
   
        return remodel.monitor_.converged, remodel.transmat_, Z_predict
    
    def dwell(self, X):
        t_ub = []
        t_bu = []
        Z = [i for i in X]
        
        # stop if only bound or unbound state
        if max(Z) == min(Z):
            return [], []
        
        for i in range(n_frame-1):
            if Z[i] == 0 and Z[i+1] == 1: # time at binding
                t_ub.append(i)
            elif Z[i] == 1 and Z[i+1] == 0: # time at unbinding 
                t_bu.append(i)
            else:
                pass
            
        # Either binding or unbinding is zero event
        if len(t_bu)*len(t_ub) == 0:
            return [], []
                   
        t_ub = np.array(t_ub)
        t_bu = np.array(t_bu)
        
        if t_ub[0] < t_bu[0]: # if binding starts first 
            t_b = t_bu - t_ub[:len(t_bu)]
            if len(t_ub) > 1:
                t_u = t_ub[1:] - t_bu[:len(t_ub[1:])]
            else: 
                return t_b, []
        else: # if unbinding starts first
            t_u = t_ub - t_bu[:len(t_ub)]
            if len(t_bu) > 1:
                t_b = t_bu[1:] - t_ub[:len(t_bu[1:])]
            else:
                return [], t_u
            
        return t_b, t_u
    
    def plot1(self, X, Z_true, Z_predict):
        
        # Sequence of true states
        X_true = np.zeros(n_frame)
        for i in range(2):
            X_true[Z_true==i] = X[Z_true==i].mean()
                
        # Sequence of predicted states
        X_predict = np.zeros(n_frame)
        for i in range(2):
            X_predict[Z_predict == i] = X[Z_predict == i].mean()        
                
        # Percent of error
        percent_error = sum(abs(Z_true - Z_predict))/n_frame*100
#        print("Prediction error = ", percent_error, "%")
        
        # Plot the sampled data      
        fig1 = plt.figure(1, clear=True)    
        
        sp1 = fig1.add_subplot(211)          
        sp1.plot(X, "k", label="observations", ms=1, lw=1, alpha=0.5)
        sp1.set_title("SNR = %.1f" % (SNR))        
         
        sp2 = fig1.add_subplot(212)         
        sp2.plot(X, "k", label="observations", ms=1, lw=1, alpha=0.5)
        sp2.plot(X_true, "b", label="states", lw=5, alpha=1)
        sp2.plot(X_predict, "r", label="predictions", lw=5, alpha=0.7)
        sp2.set_title("Error = %.1f %%" % (percent_error))
        plt.show()
        
    def plot2(self, tp_ub, tp_bu)    :
        log_tp_ub = np.log10(np.array(tp_ub))
        log_tp_bu = np.log10(np.array(tp_bu))
        
        # Remove outliers > 5*std
        log_tp_ub = reject_outliers(log_tp_ub)
        log_tp_bu = reject_outliers(log_tp_bu)        
        
        # MLE fitting with a Gaussian function
        result = MLE_G(log_tp_bu)
        m_b, s_b = result["x"]
        time_b = 1/10**(m_b) 
        print("time_b = %.1f" %(time_b)) 

        result = MLE_G(log_tp_ub)
        m_u, s_u = result["x"]
        time_u = 1/10**(m_u)
        print("time_u = %.1f" %(time_u))
                  
        fig2 = plt.figure(2, clear=True)    
        
        sp = fig2.add_subplot(121)  
        sp.hist(log_tp_bu, bins ='scott', color='k', histtype='step', density='True', lw=1)
        x = np.linspace(min(log_tp_bu), max(log_tp_bu), 100)
        sp.plot(x, Gaussian(m_b, s_b, x), 'r', lw=2)
        sp.axvline(x=m_b, color='k', linestyle='dotted', lw=1)   
        sp.set_xlabel('Log10(TP_bu)')
        sp.set_title("Bound time = %.1f" %(time_b))        
        
        sp = fig2.add_subplot(122)  
        sp.hist(log_tp_ub, bins ='scott', color='k', histtype='step', density='True', lw=1)
        x = np.linspace(min(log_tp_ub), max(log_tp_ub), 100)
        sp.plot(x, Gaussian(m_u, s_u, x), 'r', lw=2)
        sp.axvline(x=m_u, color='k', linestyle='dotted', lw=1)   
        sp.set_xlabel('Log10(TP_ub)')
        sp.set_title("Unbound time = %.1f" %(time_u))      
        plt.show()

    def plot3(self, dwell_b, dwell_u):
        dwell_b = np.array(dwell_b)
        dwell_u = np.array(dwell_u)
        
        # MLE fitting with an Exponential function
        result = MLE_E(dwell_b)
        m_b = float(result["x"])
        print("MLE m_b = %.1f" %(m_b))

        result = MLE_E(dwell_u)
        m_u = float(result["x"])
        print("MLE m_u = %.1f" %(m_u))

        # Inverse cumulative distrubution function from dwell time data  
        t_b = np.arange(max(dwell_b)+1)  
        t_u = np.arange(max(dwell_u)+1)         
        icdf_b = icdf(dwell_b, t_b)
        icdf_u = icdf(dwell_u, t_u)

        # Curve fit of the icdf
        p_b = [time_bound]
        p_u = [time_unbound]
        p_b, c_b = scipy.optimize.curve_fit(Exp1, t_b[:-10], icdf_b[:-10], p0=p_b)#, sigma = 1/icdf_b**0.5)
#        p_u, c_u = scipy.optimize.curve_fit(Exp, t_u, icdf_u, p0=[m_u, max(icdf_u), 0])#, sigma = 1/icdf_u**0.5)
        print(p_b)
#        print(p_u)
        
        # Plot the result
        fig3 = plt.figure(3, clear=True)    
        
        sp = fig3.add_subplot(241)  
        sp.hist(dwell_b, bins ='scott', color='k', histtype='step', density='True', lw=1) 
        sp.plot(t_b, Exp_pdf(m_b, t_b), 'r', lw=2)
        sp.set_xlim([0, max(dwell_b)])
        sp.set_xlabel('Frames')    
        sp.set_ylabel('Counts')
        sp.set_title('Dwell bound')
     
        sp = fig3.add_subplot(242)  
        sp.hist(dwell_b, bins ='scott', color='k', histtype='step', density='True', lw=1)
        sp.plot(t_b, Exp_pdf(m_b, t_b), 'r', lw=2)
        sp.set_xlim([0, max(dwell_b)])
        sp.set_yscale('log')
        sp.set_xlabel('Frames')    
        sp.set_ylabel('Counts')       
        sp.set_title('Dwell bound')        
        
        sp = fig3.add_subplot(243)  
        sp.plot(t_b, icdf_b, 'k.', lw=2)
        sp.plot(t_b, Exp1(p_b[0], t_b), 'r', lw=2)        
        sp.set_xlim([0, max(t_b)])
        sp.set_ylim([0, 1])
        sp.set_xlabel('Frames')    
        sp.set_ylabel('Counts')  
        sp.set_title('ICDF bound')
        
        sp = fig3.add_subplot(244)  
        sp.plot(t_b, icdf_b, 'k.', lw=2)
        sp.plot(t_b, Exp1(p_b[0], t_b), 'r', lw=2)        
        sp.set_xlim([0, max(t_b)])
        sp.set_ylim([icdf_b[-2], 1])
        sp.set_yscale('log')
        sp.set_xlabel('Frames')    
        sp.set_ylabel('Counts')  
        sp.set_title('ICDF bound')     
        
        sp = fig3.add_subplot(245)  
        sp.hist(dwell_u, bins ='scott', color='k', histtype='step', density='True', lw=1) 
        sp.plot(t_u, Exp_pdf(m_u, t_u), 'r', lw=2)
        sp.set_xlim([0, max(dwell_u)])
        sp.set_xlabel('Frames')    
        sp.set_ylabel('Counts') 
        sp.set_title('Dwell unbound')
        
        sp = fig3.add_subplot(246)  
        sp.hist(dwell_u, bins ='scott', color='k', histtype='step', density='True', lw=1)
        sp.plot(t_u, Exp_pdf(m_u, t_u), 'r', lw=2)
        sp.set_xlim([0, max(dwell_u)])
        sp.set_yscale('log')
        sp.set_xlabel('Frames')    
        sp.set_ylabel('Counts') 
        sp.set_title('Dwell unbound')            
  
        sp = fig3.add_subplot(247)  
        sp.plot(t_u, icdf_u, 'k.', lw=2)
        sp.plot(t_u, Exp1(p_u[0], t_u), 'r', lw=2)        
        sp.set_xlim([0, max(t_u)])
        sp.set_ylim([0, 1])
        sp.set_xlabel('Frames')    
        sp.set_ylabel('Counts')  
        sp.set_title('ICDF unbound')
        
        sp = fig3.add_subplot(248)  
        sp.plot(t_u, icdf_u, 'k.', lw=2)
        sp.plot(t_u, Exp1(p_u[0], t_u), 'r', lw=2)        
        sp.set_xlim([0, max(t_u)])
        sp.set_ylim([icdf_u[-2], 1])
        sp.set_yscale('log')
        sp.set_xlabel('Frames')    
        sp.set_ylabel('Counts')  
        sp.set_title('ICDF unbound')       
      
        plt.show()        
        
        

def main():
    start_time = time.time()
    
    data = Data()
    
    tp_ub_fit = []
    tp_bu_fit = []
    dwell_b = []
    dwell_u = []
    
    for i in range(n_sample):    
        X, Z_true = data.generate()
        converged, tp, Z_predict = data.predict(X)
        if converged == True:
            tp_ub_fit.append(tp[0][1])
            tp_bu_fit.append(tp[1][0])
            t_b, t_u = data.dwell(Z_predict)
            dwell_b.extend(t_b)
            dwell_u.extend(t_u)            
        
#    print("Converged percent = %.1f" % (len(tp_ub_fit)/n_sample*100))    
            
    data.plot1(X, Z_true, Z_predict)
    data.plot2(tp_ub_fit, tp_bu_fit)    
    data.plot3(dwell_b, dwell_u)
    plt.show()
    
    elapsed_time = time.time() - start_time
    
    print("Time = %d (s)" %(elapsed_time))
    

if __name__ == "__main__":
    main()

