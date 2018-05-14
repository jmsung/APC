# SiMPull simulation

from __future__ import division, print_function, absolute_import
import numpy as np
from numpy.random import rand, randn
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

np.random.seed()

n_mol = 500
n_frame = 300
n_delay = 50
corr = np.arange(1, n_frame)

t_b = 10
t_u = 20
t_t = 1/(1/t_b+1/t_u)
noise = 0.1 # percent noise
blink = 0.01 # percent per frame
nonspecific = 0.05 # percent per frame


def exp(x, a, b, c):
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
                    if j > 0:
                        I[i][j-1] = I[i][j-1] + 1
                                                
            I[i] = I[i] + noise*randn(n_frame+n_delay)
            
            I[i] = I[i] - np.min(I[i])
            I[i] = I[i]/np.max(I[i])

            bg_u = np.mean(I[i][I[i] < 0.5])
            bg_b = np.mean(I[i][I[i] > 0.5])
            I[i] = (I[i] - bg_u)/(bg_b - bg_u) 
        self.I = I[:,n_delay:]
            

    def analyze(self):
        self.corr_b = np.zeros((n_mol, n_frame-1))
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
            
        fig3 = plt.figure(3)        
        sp1 = fig3.add_subplot(121)
        p1, pcov1 = curve_fit(exp, corr, self.corr_b_mean, p0=[0, 1, 10], sigma=self.corr_b_sem)  
        x_fit = np.linspace(0,max(corr),1000)
        y_fit = exp(x_fit, p1[0], p1[1], p1[2])         

        sp1.errorbar(corr, self.corr_b_mean, yerr=self.corr_b_sem, fmt='ko')
        sp1.plot(x_fit, y_fit, 'r', linewidth=2)        
        title = "Dwell time = %.1f (given), %.1f +/- %.1f (est)" % (t_t, p1[2], pcov1[2,2]**0.5)
        sp1.set_title(title)
        
        sp2 = fig3.add_subplot(122)
        lim = 5*p1[2]
        index = corr<lim
        p2, pcov2 = curve_fit(exp2, corr[index], self.corr_b_mean[index], p0=[p1[0], p1[1], p1[2], p1[1]/2, p1[2]/5], sigma=self.corr_b_sem[index])    
        x_fit = np.linspace(0,lim,1000)
        y_fit = exp2(x_fit, p2[0], p2[1], p2[2], p2[3], p2[4])

        sp2.errorbar(corr[corr<lim], self.corr_b_mean[corr<lim], yerr=self.corr_b_sem[corr<lim], fmt='ko')
        sp2.plot(x_fit, y_fit, 'r', linewidth=2)        
        title = "Dwell time = %.1f (given), %.1f +/- %.1f (est)" % (t_t, p2[2], (pcov2[2,2])**0.5)
        sp2.set_title(title)
                 
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