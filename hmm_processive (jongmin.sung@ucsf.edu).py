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

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import scipy 
from hmmlearn import hmm
import time


# User parameters
n_sample = 2
n_frame = 200
SNR = 10
time_bound = 10
time_unbound = 20

print("Number of sample = %d" %(n_sample))
print("Number of frame = %d" %(n_frame))
print("SNR = %d\n" %(SNR))
print("Time bound (true) = %.1f" %(time_bound))
print("Time unbound (true) = %.1f\n" %(time_unbound))


def icdf(data, time):
    data = np.array(data)
    cdf = np.zeros(len(time))
    for i in time:
        cdf[i] = sum(data <= time[i])
    icdf = 1 - cdf/max(cdf)
    icdf = icdf - min(icdf) + 0.001
    return icdf

def outliers(data, m = 5.):
    data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return s > m

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
    return abs(a)*np.exp(-x/abs(m)) + b 

def Exp2(m, b, x):
    return np.exp(-x/abs(m)) + b 

def Exp1(m, x):
    return np.exp(-x/abs(m))

def Exp_pdf(m, x):
    return np.exp(-x/abs(m))/abs(m) 

def LL_E(param, x):      
    [m] = param
    return np.sum(np.log(Exp_pdf(m, x)))  

def MLE_E(x):
    m = np.mean(x)
    fun = lambda *args: -LL_E(*args)
    p0 = [m]
    result = scipy.optimize.minimize(fun, p0, args=(x)) 
    return result
   
def get_dwell(X):
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


class Data:
    # Initialize an object with the parameters
    def __init__(self, n_sample, n_frame, SNR, time_bound, time_unbound):
        self.n_sample = n_sample 
        self.n_frame = n_frame
        self.SNR = SNR
        self.time_bound = time_bound 
        self.time_unbound = time_unbound


    # Build a HMM model and simulate n_sample traces
    def generate(self):       
        # Build an HMM instance and set parameters
        self.model = hmm.GaussianHMM(n_components=2, covariance_type="full")

        # The transition probability matrix
        tp_ub = 1/self.time_unbound
        tp_uu = 1 - tp_ub
        tp_bu = 1/self.time_bound
        tp_bb = 1 - tp_bu
 
        # Set the parameters to generate samples
        self.model.startprob_ = np.array([0.5, 0.5])
        self.model.transmat_ = np.array([[tp_uu, tp_ub],
                                         [tp_bu, tp_bb]])
        self.model.means_ = np.array([[0], [1]])
        self.model.covars_ = np.tile(np.identity(1), (2, 1, 1)) / self.SNR**2      
        
        # Generate list of n_samples and concatenate them
        self.X = [None] * self.n_sample
        self.Z_true = [None] * self.n_sample
        self.X_conc = np.empty(shape=(0, 1)) 
        self.Z_true_conc = np.empty(shape=(0, 1)) 
        for i in range(self.n_sample):
            self.X[i], Z = self.model.sample(self.n_frame)
            self.Z_true[i] = Z.reshape(self.n_frame, 1)            
            self.X_conc = np.concatenate((self.X_conc, self.X[i]), axis=0)
            self.Z_true_conc = np.concatenate((self.Z_true_conc, self.Z_true[i]), axis=0)


    # HMM prediction       
    def predict(self):    
        # Set a new model for traidning
        self.remodel = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)        
        
        # Set initial parameters for training
        self.remodel.startprob_ = np.array([0.5, 0.5])
        self.remodel.transmat_ = np.array([[0.5, 0.5], 
                                      [0.5, 0.5]])
        self.remodel.means_ = np.array([0, 1])
        self.remodel.covars_ = np.tile(np.identity(1), (2, 1, 1)) / self.SNR**2    

        self.Z_predict = [None] * self.n_sample
        self.converged = [None] * self.n_sample
        self.X_mean = [None] * self.n_sample
        self.X_var = [None] * self.n_sample    
        self.SNR = np.zeros(self.n_sample)
        self.tp = [None] * self.n_sample 
        self.tp_ub = np.zeros(self.n_sample) 
        self.tp_bu = np.zeros(self.n_sample) 
        self.tb_HMM = np.zeros(self.n_sample)
        self.tu_HMM = np.zeros(self.n_sample)  
        
        for i in range(n_sample):              
            # Estimate model parameters (training)
            self.remodel.fit(self.X[i])  
        
            # Find most likely state sequence corresponding to X
            Z_predict = self.remodel.predict(self.X[i])
            Z_predict = Z_predict.reshape(self.n_frame, 1)  
            X_mean = self.remodel.means_ # Mean  
            X_var = self.remodel.covars_ # Covariance   
                      
### Simplify the following 
            tp = self.remodel.transmat_ # Transition probability                 
            self.converged[i] = self.remodel.monitor_.converged # Check convergence
            self.SNR[i] = (abs(X_mean[1][0]-X_mean[0][0])/(np.mean(X_var))**0.5)            

            # Assign them such that X[state==0]=0 and X[state==1]=1
            if X_mean[0] <= X_mean[1]:  
                self.Z_predict[i] = Z_predict
                self.X_mean[i] = [X_mean[0][0], X_mean[1][0]]
                self.X_var[i] = [X_var[0][0][0], X_var[1][0][0]]             
                self.tp[i] = [[tp[0][0], tp[0][1]],
                              [tp[1][0], tp[1][1]]]
            else:     
                self.Z_predict[i] = 1 - Z_predict 
                self.X_mean[i] = [X_mean[1][0], X_mean[0][0]]
                self.X_var[i] = [X_var[1][0][0], X_var[0][0][0]]           
                self.tp[i] = [[tp[1][1], tp[1][0]],
                              [tp[0][1], tp[0][0]]]
  
            # HMM estimate of bound (tb) and unbound time (tu)
            self.tp_ub[i] = self.tp[i][0][1] + 1/n_frame # Transition prob from unbound to bound
            self.tp_bu[i] = self.tp[i][1][0] + 1/n_frame # Transition prob from bound to unbound
            self.tb_HMM[i] = 1/self.tp_bu[i] # Bound time
            self.tu_HMM[i] = 1/self.tp_ub[i] # Unbound time
      
        # Check the convergence
        print("%.1f %% converged." %(sum([int(i) for i in self.converged])/self.n_sample*100))

        # Label only good data
        cond1 = np.array(self.tb_HMM) <= n_frame*0.5
#        cond1 = ~outliers(self.tb_HMM)
        cond2 = np.array(self.tu_HMM) <= n_frame*0.5  
#        cond2 = ~outliers(self.tu_HMM)
        cond3 = ~outliers(self.SNR)
        self.good_data = cond1 & cond2 & cond3

        # Log transition probability
        self.log_tp_ub = np.log10(np.array(self.tp_ub[self.good_data]))
        self.log_tp_bu = np.log10(np.array(self.tp_bu[self.good_data]))              
        
        # MLE fitting with a Gaussian function
        result_bu = MLE_G(self.log_tp_bu)
        result_ub = MLE_G(self.log_tp_ub)               
        self.m_b, self.s_b = result_bu["x"]
        self.m_u, self.s_u = result_ub["x"]               
        self.tb_MLE = 1/10**(self.m_b) 
        self.tu_MLE = 1/10**(self.m_u)           
        error_tb = 100*(self.tb_MLE/self.time_bound-1)
        error_tu = 100*(self.tu_MLE/self.time_unbound-1)  
        print("Time bound (MLE) = %.1f (%.1f %%)" %(self.tb_MLE, error_tb)) 
        print("Time unbound (MLE) = %.1f (%.1f %%) \n" %(self.tu_MLE, error_tu)) 

#        # ----------------------------------------------------------------------
#        # HMM prediction with concatenated data   
#        self.remodel.fit(self.X_conc) # Fit (train) to find the parameters         
#        Z_predict_conc = self.remodel.predict(self.X_conc) # Predict the most likely trajectory
#        self.Z_predict_conc = Z_predict_conc.reshape(self.n_frame*self.n_sample, 1)          
#        self.converged_conc = self.remodel.monitor_.converged # Check the convergence
#        self.tp_conc = self.remodel.transmat_ # Transition probability

#        # Reorder state number such that X[Z=0] < X[Z=1] 
#        if self.X_conc[Z_predict_conc == 0].mean() > self.X_conc[Z_predict_conc == 1].mean():
#            self.Z_predict_conc = 1 - self.Z_predict_conc
#            self.tp_conc = np.array([[self.tp_conc[1][1], self.tp_conc[1][0]],
#                                     [self.tp_conc[0][1], self.tp_conc[0][0]]])#
#
#        self.tp_bu_conc = self.tp_conc[1][0] + 1/n_frame # Transition prob from unbound to bound
#        self.tp_ub_conc = self.tp_conc[0][1] + 1/n_frame # Transition prob from bound to unbound
#        self.tb_HMM_conc = 1/self.tp_bu_conc # Bound time
#        self.tu_HMM_conc = 1/self.tp_ub_conc # Unbound time
#        error_tb = 100*(self.tb_HMM_conc/self.time_bound-1)
#        error_tu = 100*(self.tu_HMM_conc/self.time_unbound-1)      
#        print("HMM_concatenated is %s" %(["not converged.", "converged."][int(self.converged_conc)]))
#        print("Time bound (HMM, conc) = %.1f (%.1f %%)" %(self.tb_HMM_conc, error_tb))
#        print("Time unbound (HMM, conc) = %.1f (%.1f %%)\n" %(self.tu_HMM_conc, error_tu))


    def threshold(self):
        self.Z_Threshold = [None] * self.n_sample
        self.X_Threshold = [None] * self.n_sample            
  
        for i in range(self.n_sample):
            self.Z_Threshold[i] = np.zeros(self.n_frame)
            for j in range(self.n_frame):
                if self.X[i][j] > 0.5:
                    self.Z_Threshold[i][j] = 1

#            self.Z_Threshold[i][self.X[i]>0.5] = 1
#            self.X_Threshold[i][self.Z_Threshold[i]==0] = np.mean(self.X[i][self.Z_Threshold[i]==0])
#            self.X_Threshold[i][self.Z_Threshold[i]==1] = np.mean(self.X[i][self.Z_Threshold[i]==1])            
      
        
        
    def analyze_pdf(self):

        dwell_b = []
        dwell_u = []
        
        for i in range(self.n_sample):      
            if self.good_data[i]:
                tb, tu = get_dwell(self.Z_predict[i])
                dwell_b.extend(tb)                   
                dwell_u.extend(tu)
        
        dwell_b = np.array(dwell_b)
        dwell_u = np.array(dwell_u)
        
        self.dwell_b = dwell_b[dwell_b < self.n_frame*0.5]
        self.dwell_u = dwell_u[dwell_u < self.n_frame*0.5]
        
        self.dwell_b_min = 0#np.min(self.dwell_b)
        self.dwell_u_min = 0#np.min(self.dwell_u)        
               
        # MLE fitting with an Exponential function
        result_b = MLE_E(self.dwell_b - self.dwell_b_min)
        result_u = MLE_E(self.dwell_u - self.dwell_u_min)        
        self.tb_pdf = float(result_b["x"]) + self.dwell_b_min
        self.tu_pdf = float(result_u["x"]) + self.dwell_u_min       
        error_tb = 100*(self.tb_pdf/self.time_bound-1)
        error_tu = 100*(self.tu_pdf/self.time_unbound-1) 
        print("Time bound (PDF) = %.1f (%.1f %%)" %(self.tb_pdf, error_tb))
        print("Time unbound (PDF) = %.1f (%.1f %%) \n" %(self.tu_pdf, error_tu))

       
    def analyze_icdf(self):
        
        self.t_b = np.arange(max(self.dwell_b))  
        self.t_u = np.arange(max(self.dwell_u))   

        # Inverse cumulative distrubution function from dwell time data         
        self.icdf_b = icdf(self.dwell_b, self.t_b)
        self.icdf_u = icdf(self.dwell_u, self.t_u)

        p_b = [self.tb_pdf, 1, 0]
        p_u = [self.tu_pdf, 1, 0]

        # Curve fit of the icdf
#        p_b, c_b = scipy.optimize.curve_fit(Exp3, t_b[10:30], icdf_b[10:30], p0=p_b)#, [m_b, 1, 0])#, sigma = 1/icdf_b**0.5)
#        p_u, c_u = scipy.optimize.curve_fit(Exp3, t_u[10:30], icdf_u[10:30], p0=p_u)#, sigma = 1/icdf_u**0.5)
 
        self.p_b, self.c_b = scipy.optimize.curve_fit(lambda t,a,b,c: b*np.exp(-t/a)+c, self.t_b, self.icdf_b, p0=p_b)#, sigma=1/icdf_b**0.5)
        self.p_u, self.c_u = scipy.optimize.curve_fit(lambda t,a,b,c: b*np.exp(-t/a)+c, self.t_u, self.icdf_u, p0=p_u)#, sigma=1/icdf_u**0.5)

        print('Time bound (icdf) = %.1f (%.1f %%)' %(self.p_b[0], 100*(self.p_b[0]/time_bound-1)))  
        print('Time unbound (icdf) = %.1f (%.1f %%) \n' %(self.p_u[0], 100*(self.p_u[0]/time_unbound-1)))  
              
    
    def plot_trace(self):

        fig = plt.figure('trace', clear=True) 
        
        for i in range(1):
            # Mean values for true and predicted states
            X_true = np.zeros((n_frame,1))
            X_predict = np.zeros((n_frame,1))        
            X_threshold = np.zeros((n_frame,1)) 
            
            for j in range(2):
                X_true[self.Z_true[i]==j] = self.X[i][self.Z_true[i]==j].mean()
                X_predict[self.Z_predict[i]==j] = self.X[i][self.Z_predict[i]==j].mean()   
                X_threshold[self.Z_Threshold[i]==j] = self.X[i][self.Z_Threshold[i]==j].mean()      
                    
            # Percent of error
            percent_error = sum(abs(self.Z_true[i] - self.Z_predict[i]))/self.n_frame*100

            # Plot the sampled data
            sp = fig.add_subplot(2, 4, i+1)         
            sp.plot(self.X[i], "k", label="observations", ms=1, lw=1, alpha=0.5)
            sp.plot(X_true, "b", label="states", lw=2, alpha=1)
            sp.plot(X_predict, "r", label="predictions", lw=2, alpha=1)
            sp.set_ylabel('Intensity')
#            sp.set_title("SNR = %.1f, HMM" % (self.SNR[i])) #show both signal and noise
            sp.set_title("SNR = %.1f, Error = %.1f %%" % (self.SNR[i], percent_error)) #show both signal and noise
            
#            sp = fig.add_subplot(2, 1, i+2)         
#            sp.plot(self.X[i], "k", label="observations", ms=1, lw=1, alpha=0.5)
#            sp.plot(X_true, "b", label="states", lw=2, alpha=1)
#            sp.plot(X_threshold, "r", label="threshold", lw=2, alpha=1)       
#            sp.set_xlabel('Frame')
#            sp.set_ylabel('Intensity')      
#            sp.set_title("SNR = %.1f, Threshoding" % (self.SNR[i])) #show both signal and noise            
            
            
#            sp.set_title("SNR = %.1f, Error = %.1f %%" % (self.SNR[i], percent_error)) #show both signal and noise
            plt.show()


    def plot_cluster(self):
        fig = plt.figure('cluster', clear=True)      
        
        sp = fig.add_subplot(231)
        sp.hist(self.SNR, bins ='scott', color='k', histtype='step', lw=1)  
        sp.hist(self.SNR[self.good_data], bins ='scott', color='r', histtype='step', lw=1)         
        sp.set_title('SNR')
        
        sp = fig.add_subplot(232)
        sp.hist(self.tb_HMM, bins ='scott', color='k', histtype='step', lw=1)  
        sp.hist(self.tb_HMM[self.good_data], bins ='scott', color='r', histtype='step', lw=1)         
        sp.set_title('Time bound (True = %.1f)' %(self.time_bound))
        
        sp = fig.add_subplot(233)
        sp.hist(self.tu_HMM, bins ='scott', color='k', histtype='step', lw=1)     
        sp.hist(self.tu_HMM[self.good_data], bins ='scott', color='r', histtype='step', lw=1)         
        sp.set_title('Time unbound (True = %.1f)' %(self.time_unbound))        
        
        sp = fig.add_subplot(234)
        sp.plot(self.SNR, self.tb_HMM, 'k.', alpha=0.5)
        sp.plot(self.SNR[self.good_data], self.tb_HMM[self.good_data], 'r.', alpha=0.5)        
        sp.set_xlabel('SNR')
        sp.set_ylabel('Time bound')       
        
        sp = fig.add_subplot(235)
        sp.plot(self.SNR, self.tu_HMM, 'k.', alpha=0.5)
        sp.plot(self.SNR[self.good_data], self.tu_HMM[self.good_data], 'r.', alpha=0.5)             
        sp.set_xlabel('SNR')          
        sp.set_ylabel('Time unbound')
                 
        sp = fig.add_subplot(236)
        sp.plot(self.tb_HMM, self.tu_HMM, 'k.', alpha=0.5)
        sp.plot(self.tb_HMM[self.good_data], self.tu_HMM[self.good_data], 'r.', alpha=0.5)              
        sp.set_xlabel('Time bound')
        sp.set_ylabel('Time unbound')      
        plt.show()        

        
    def plot_HMM(self):
                
        fig = plt.figure('HMM', clear=True)    
        
        sp = fig.add_subplot(121)  
        sp.hist(self.log_tp_bu, bins ='scott', color='k', histtype='step', density='True', lw=1)
        x = np.linspace(min(self.log_tp_bu), max(self.log_tp_bu), 100)
        sp.plot(x, Gaussian(self.m_b, self.s_b, x), 'r', lw=2)
        sp.axvline(x=self.m_b, color='k', linestyle='dotted', lw=1)   
        sp.set_xlabel('Log10(TP_bu)')
        error_b = 100*(self.tb_MLE/time_bound-1)
        sp.set_title("Bound time = %.1f (%.1f %%)" %(self.tb_MLE, error_b))        
        
        sp = fig.add_subplot(122)  
        sp.hist(self.log_tp_ub, bins ='scott', color='k', histtype='step', density='True', lw=1)
        x = np.linspace(min(self.log_tp_ub), max(self.log_tp_ub), 100)
        sp.plot(x, Gaussian(self.m_u, self.s_u, x), 'r', lw=2)
        sp.axvline(x=self.m_u, color='k', linestyle='dotted', lw=1)   
        sp.set_xlabel('Log10(TP_ub)')
        error_u = 100*(self.tu_MLE/time_unbound-1)
        sp.set_title("Unbound time = %.1f (%.1f %%)" %(self.tu_MLE, error_u))      
        plt.show()


    def plot_pdf(self):
      
        # Plot the result
        fig = plt.figure('PDF', clear=True)    
        
        sp = fig.add_subplot(121)  
        sp.hist(self.dwell_b - self.dwell_b_min, bins=np.linspace(0,max(self.dwell_b),20), color='k', histtype='step', density='True', lw=1) 
        sp.plot(self.t_b, Exp_pdf(self.tb_pdf, self.t_b), 'r', lw=1)
        sp.set_xlim([0, max(self.dwell_b)])
        sp.set_xlabel('Frames')    
        sp.set_ylabel('Probability')
        error_b = 100*(self.tb_pdf/time_bound-1)
        sp.set_title('Time bound (PDF) = %.1f (%.1f %%)' %(self.tb_pdf, error_b))          
         
        sp = fig.add_subplot(122)  
        sp.hist(self.dwell_u - self.dwell_u_min, bins=np.linspace(0,max(self.dwell_u),20), color='k', histtype='step', density='True', lw=1) 
        sp.plot(self.t_u, Exp_pdf(self.tu_pdf, self.t_u), 'r', lw=1)
        sp.set_xlim([0, max(self.dwell_u)])
        sp.set_xlabel('Frames')    
        sp.set_ylabel('Probability')
        error_u = 100*(self.tu_pdf/time_unbound-1)
        sp.set_title('Time unbound (PDF) = %.1f (%.1f %%)' %(self.tu_pdf, error_u))  
          
        plt.show()        
        
    def plot_icdf(self):
        
        fig = plt.figure('ICDF', clear=True)    
        
        sp = fig.add_subplot(121)  
        sp.plot(self.t_b, self.icdf_b, 'ko', ms=2)
        sp.plot(self.t_b, Exp3(self.p_b[0], self.p_b[1], self.p_b[2], self.t_b), 'r', lw=1)        
        sp.set_xlim([0, max(self.t_b)])
        sp.set_ylim([0, 1])
        sp.set_xlabel('Frames')    
        sp.set_ylabel('Pobability')  
        sp.set_title('Time bound (icdf) = %.1f (%.1f %%)' %(self.p_b[0], 100*(self.p_b[0]/self.time_bound-1)))    
                   
        
        sp = fig.add_subplot(122)  
        sp.plot(self.t_u, self.icdf_u, 'ko', ms=2)
        sp.plot(self.t_u, Exp3(self.p_u[0], self.p_u[1], self.p_u[2], self.t_u), 'r', lw=1)        
        sp.set_xlim([0, max(self.t_u)])
        sp.set_ylim([0, 1])        
        sp.set_xlabel('Frames')    
        sp.set_ylabel('Probability')  
        sp.set_title('Time unbound (icdf) = %.1f (%.1f %%)' %(self.p_u[0], 100*(self.p_u[0]/self.time_unbound-1)))    
      
    
          

def main(n_sample, n_frame, SNR, time_bound, time_unbound):
    start_time = time.time()
    
    # Initialize 
    data = Data(n_sample, n_frame, SNR, time_bound, time_unbound)
    
    # Generate list of data array
    data.generate()
    
    # Predict from list of data array
    data.predict()  
  
    # Test thresholding 
    data.threshold()
    
    # PDF analysis
#    data.analyze_pdf()
    
    # ICDF analysis
#    data.analyze_icdf()
    
    # Plot the result
    data.plot_trace()
#    data.plot_cluster()
    data.plot_HMM()     
    data.plot_pdf()
#    data.plot_icdf()    
       
    print("Calculation time = %d (s)" %(time.time()-start_time))
    

if __name__ == "__main__":
    main(n_sample, n_frame, SNR, time_bound, time_unbound)


# To do
# Independent signal generation
# Cutoff > Median estimat
# TP = Beta distribution


