# SiMPull simulation

from __future__ import division, print_function, absolute_import
import numpy as np
from numpy.random import rand, randn
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

np.random.seed()

n_mol = 100
n_frame = 100
n_delay = 50
n_corr = n_frame-1
corr = np.arange(1, n_corr)

t_b = 10
t_u = 5
t_t = 1/(1/t_b+1/t_u)
noise = 0.2 # percent noise
blink = 0 # percent per frame
nonspecific = 0 # percent per frame


def exp1(x, a, b, c):
    return a + b * np.exp(-x/c) 

def exp2(x, a, b, c, d, e):
    return a + b * np.exp(-x/c) + d* np.exp(-x/e) 

class Data(object):
    def __init__(self):
        pass
        
    def generate(self):
        I = np.zeros((n_mol, n_frame+n_delay))
        
        for i in range(n_mol):
            for j in range(n_frame+n_delay-1):
                if I[i][j] == 0:
                    if rand() < 1 - np.exp(-1/t_u):
                        I[i][j+1] = 1
                else:
                    if rand() > 1 - np.exp(-1/t_b):
                        I[i][j+1] = 1         
                    if rand() < blink:
                        I[i][j] = 0
                if rand() < nonspecific:
                    I[i][j] = I[i][j] + 1   
                                                
            I[i] = I[i] + noise*randn(n_frame+n_delay)
            
            I[i] = I[i] - np.min(I[i])
            I[i] = I[i]/np.max(I[i])

            bg_u = np.mean(I[i][I[i] < 0.5])
            bg_b = np.mean(I[i][I[i] > 0.5])
            I[i] = (I[i] - bg_u)/(bg_b - bg_u) 
        self.I = I[:,n_delay:]
            

    def analyze(self):
        self.corr_b = np.zeros((n_mol, n_corr-1))
        for i in range(n_mol):
            for j in corr:
                corr_b = []
                for k in range(n_frame-j):
                    corr_b.append(self.I[i][k]*self.I[i][k+j])
                self.corr_b[i, j-1] = np.mean(corr_b)
        self.corr_b_mean = np.mean(self.corr_b, axis=0)
        self.corr_b_sem = np.std(self.corr_b, axis=0)/n_mol**0.5


    def plot(self):
        # Figure 1
        fig1 = plt.figure(1)  
        row = 5
        col = 4  
        for i in range(row*col):        
            sp = fig1.add_subplot(row, col, i+1)  
            sp.plot(self.I[i], 'k-')
            sp.axhline(y=0, color='b', linestyle='dashed', linewidth=1)
            sp.axhline(y=1, color='b', linestyle='dashed', linewidth=1)
            
        fig2 = plt.figure(2)
        for i in range(row*col):        
            sp = fig2.add_subplot(row, col, i+1)  
            sp.plot(self.corr_b[i], 'k-')    
            sp.axhline(y=0, color='b', linestyle='dashed', linewidth=1) 
            
        # Figure 3. Correlation - Full range
        fig3 = plt.figure(3)  
        p0 = [0, 1, 10]
        p1, pcov1 = curve_fit(exp1, corr, self.corr_b_mean, p0, self.corr_b_sem) 
        x_fit = np.linspace(0,max(corr),1000)
        y_fit1 = exp1(x_fit, p1[0], p1[1], p1[2])        
        scale1 = y_fit1[0]
        offset1 = p1[0]
        y_fit1 = (y_fit1 - offset1)/(scale1 - offset1)
        self.corr_b_mean1 = (self.corr_b_mean - offset1)/(scale1 - offset1)    

        sp1 = fig3.add_subplot(121)
        sp1.plot(corr, self.corr_b_mean1, 'ko', mfc='none')
        sp1.plot(x_fit, y_fit1, 'r', linewidth=2)     
        sp1.set_xlim([0, max(corr)])  
        sp1.set_ylim([-0.1, 1])     
        title = "[Given] t_b = %.1f, t_u = %.1f, t_tot = %.1f " % (t_b, t_u, t_t)
        sp1.set_title(title)
        sp1.set_xlabel('Lag time [Frame]')
        sp1.set_ylabel('Correlation [AU]')
        
        sp2 = fig3.add_subplot(122)
        sp2.plot(corr, self.corr_b_mean1, 'ko', mfc='none')
        sp2.set_yscale('log')
        sp2.semilogy(x_fit, y_fit1, 'r', linewidth=2)    
        sp2.set_xlim([0, max(corr)])  
        sp2.set_ylim([min(y_fit1)/10, 1])          
        title = "[Estimate] t_est = %.1f +/- %.1f" % (p1[2], pcov1[2,2]**0.5)
        sp2.set_title(title)
        sp2.set_xlabel('Lag time [frame]')
        sp2.set_ylabel('Correlation [AU]')
  
        # Figure 4. Correlation - Short range
        fig4 = plt.figure(4)  
        p0 = [p1[0], p1[1], p1[2]]
        lim = corr < p1[2]*2
        p1, pcov1 = curve_fit(exp1, corr[lim], self.corr_b_mean[lim], p0, self.corr_b_sem[lim]) 
        x_fit = np.linspace(0,max(corr[lim]),1000)
        y_fit1 = exp1(x_fit, p1[0], p1[1], p1[2])        
        scale1 = y_fit1[0]
        offset1 = p1[0]
        y_fit1 = (y_fit1 - offset1)/(scale1 - offset1)
        self.corr_b_mean1 = (self.corr_b_mean - offset1)/(scale1 - offset1)    

        sp1 = fig4.add_subplot(121)
        sp1.plot(corr[lim], self.corr_b_mean1[lim], 'ko', mfc='none')
        sp1.plot(x_fit, y_fit1, 'r', linewidth=2)     
        sp1.set_xlim([0, max(corr[lim])])  
        sp1.set_ylim([0, 1])     
        title = "[Given] t_b = %.1f, t_u = %.1f, t_tot = %.1f " % (t_b, t_u, t_t)
        sp1.set_title(title)
        sp1.set_xlabel('Lag time [Frame]')
        sp1.set_ylabel('Correlation [AU]')
        
        sp2 = fig4.add_subplot(122)
        sp2.plot(corr[lim], self.corr_b_mean1[lim], 'ko', mfc='none')
        sp2.set_yscale('log')
        sp2.semilogy(x_fit, y_fit1, 'r', linewidth=2)    
        sp2.set_xlim([0, max(corr[lim])])  
        sp2.set_ylim([min(y_fit1)/2, 1])          
        title = "[Estimate] t_est = %.1f +/- %.1f" % (p1[2], pcov1[2,2]**0.5)
        sp2.set_title(title)
        sp2.set_xlabel('Lag time [frame]')
        sp2.set_ylabel('Correlation [AU]')                 
                                               
        plt.show()
      
# Start  
plt.close('all')
data = Data()
print('Generating...')
data.generate()
print('Analyzing...')
data.analyze()
print('Plotting...')
data.plot()