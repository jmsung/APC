"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Created by Jongmin Sung (jongmin.sung@gmail.com)

Anisotropy data analysis 

The equation for the curve as published by Marchand et al. in Nature Cell Biology in 2001 is as follows:
y = a + (b-a) / [(c(x+K)/K*d)+1], where 
a is the anisotropy without protein,
b is anisotropy with protein,
c is the Kd for ligand, 
d is the total concentration of protein. 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path  
import os
import shutil
from timeit import default_timer as timer
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from inspect import currentframe, getframeinfo
fname = getframeinfo(currentframe()).filename # current file name
current_dir = Path(fname).resolve().parent

# User input ----------------------------------------------------------------
red_x = np.array([100, 50, 25, 12.5, 6.25, 3.125, 1.56, 0])
red_y = np.array([0.179, 0.186, 0.19, 0.195, 0.2, 0.212, 0.222, 0.248])
red_p = np.array([0.191, 0.248, 0.05, 1])

black_x = np.array([100, 50, 25, 18.75, 14.1, 10.5, 7.9, 5.9, 0])
black_y = np.array([0.204, 0.225, 0.248, 0.26, 0.268, 0.271, 0.274, 0.277, 0.278])
black_p = np.array([0.183, 0.278, 1.5, 16])

# ---------------------------------------------------------------------------
def red_anisotropy(x, K):
    a = red_p[0]
    b = red_p[1]
    c = red_p[2]
    d = red_p[3]
    return a+(b-a)/((c*(x+K)/(K*d))+1)

def black_anisotropy(x, K):
    a = black_p[0]
    b = black_p[1]
    c = black_p[2]
    d = black_p[3]
    return a+(b-a)/((c*(x+K)/(K*d))+1)    


def main():
    red_p, _ = curve_fit(red_anisotropy, red_x, red_y, p0=[0.078])
    black_p, _ = curve_fit(black_anisotropy, black_x, black_y, p0=[0.1])

    # Plot the result 

    fit_x = np.linspace(0, 100, 1000)

    fig, (ax1, ax2) = plt.subplots(figsize=(20, 10), ncols=2, nrows=1, dpi=300)
       
    ax1.plot(red_x, red_y, 'ro', ms=10)        
    ax1.plot(fit_x, red_anisotropy(fit_x, red_p), 'r', lw=2)     
    ax1.set_xlabel('[dark D] um')
    ax1.set_ylabel('Anisotropy')
    ax1.set_title('Red K = %f' %(red_p))
    ax1.set_ylim([0.15, 0.3])
       
    ax2.plot(black_x, black_y, 'ko', ms=10)        
    ax2.plot(fit_x, black_anisotropy(fit_x, black_p), 'k', lw=2)     
    ax2.set_xlabel('[dark D] um')
    ax2.set_ylabel('Anisotropy')
    ax2.set_title('Black K = %f' %(black_p))
    ax2.set_ylim([0.15, 0.3])

    fig.savefig('plot_anisotropy.png')   
    plt.close(fig)

if __name__ == "__main__":
    main()


