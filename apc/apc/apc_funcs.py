# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:30:05 2019

@author: Jongmin Sung

custom_funcs.py
In projectname/projectname/custom_funcs.py, we can put in custom code that 
gets used across more than notebook. One example would be downstream data 
preprocessing that is only necessary for a subset of notebooks.

# custom_funcs.py

"""
from __future__ import division, print_function, absolute_import
import numpy as np
import random
import os
from imreg_dft.imreg import translation
from scipy.optimize import minimize
from skimage import filters
from PIL import Image
from tifffile import TiffFile
import csv
from scipy import optimize
from hmmlearn import hmm
#import pandas as pd

def read_movie1(movie_path):
    movie = Image.open(movie_path)
    n_frame = movie.n_frames
    n_row = movie.size[1]
    n_col = movie.size[0]

    # Read tif file and save into I[frame, row, col]
    I = np.zeros((n_frame, n_row, n_col), dtype=int)
    for i in range(n_frame): 
        movie.seek(i) # Move to i-th frame
        I[i,] = np.array(movie, dtype=int)

    m = 100
    n_row = int(np.floor(n_row/m)*m)
    n_col = int(np.floor(n_col/m)*m) 

    return I[:,:n_row,:n_col]

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def read_movie(movie_path, bin_size):
    # read tiff file
    with TiffFile(movie_path) as tif:
        imagej_hyperstack = tif.asarray()
        imagej_metadata = str(tif.imagej_metadata)
        imagej_metadata = imagej_metadata.split(',')

    n_frame = np.size(imagej_hyperstack, 0)
    n_row = np.size(imagej_hyperstack, 1)
    n_col = np.size(imagej_hyperstack, 2) 

    # write meta_data    
    with open(movie_path.parent/'meta_data.txt', 'w') as f:
        for item in imagej_metadata:
            f.write(item+'\n')

    # Crop the image to make the size integer multiple of 10
    m = bin_size
    n_row = int(int(n_row/m)*m)
    n_col = int(int(n_col/m)*m)
    imagej_crop = imagej_hyperstack[:,:n_row,:n_col]
    print('[frame, row, col] = [%d, %d, %d]' %(n_frame, n_row, n_col))  

    return imagej_crop, imagej_metadata

def gaussian_2d(height, center_x, center_y, width_x, width_y, offset):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    offset = float(offset)
    return lambda x,y: height*np.exp(                
                - (((center_x - (x))/width_x)**2     
                  +((center_y - (y))/width_y)**2)/2) + offset

def moments(data):
    """Returns (height, x, y, width_x, width_y, offset)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    offset = data.min()
    return height, x, y, width_x, width_y, offset

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian_2d(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p

def flatfield_correct(I, bin_size):
    n_frame = np.size(I, 0)
    n_row = np.size(I, 1)
    n_col = np.size(I, 2)

    # Binning
    I_max = np.max(I, axis=0)   
    I_bin = np.array(I_max)
    m = bin_size    
    for i in range(int(n_row/m)):
        for j in range(int(n_col/m)):
            I_bin[i*m:(i+1)*m, j*m:(j+1)*m] = np.median(I_max[i*m:(i+1)*m, j*m:(j+1)*m])

    # Gaussian fitting to the bin
    """ param = height, x, y, width_x, width_y, offset """
    params = fitgaussian(I_bin)
    I_fit = gaussian_2d(*params)(*np.indices(I_max.shape))

    # Flatfield correct
    I_flatfield = np.array(I)
    for i in range(n_frame):
        I_flatfield[i,] = I[i,] / I_fit * np.max(I_fit)

    I_flat_max = np.max(I_flatfield, axis=0)   
    I_flat_bin = np.array(I_max)
    for i in range(int(n_row/m)):
        for j in range(int(n_col/m)):
            I_flat_bin[i*m:(i+1)*m, j*m:(j+1)*m] = np.median(I_flat_max[i*m:(i+1)*m, j*m:(j+1)*m])

    return I_bin, I_fit, I_flatfield, I_flat_bin


def drift_correct(I):
    I_ref = I[int(len(I)/2),] # Mid frame as a reference frame
#    I_ref = np.max(I, axis=0)

    # Translation as compared with I_ref
    d_row = np.zeros(len(I), dtype='int')
    d_col = np.zeros(len(I), dtype='int')
    for i, I_frame in enumerate(I):
        result = translation(I_ref, I_frame)
        d_row[i] = round(result['tvec'][0])
        d_col[i] = round(result['tvec'][1])      

    # Changes of translation between the consecutive frames
    dd_row = d_row[1:] - d_row[:-1]
    dd_col = d_col[1:] - d_col[:-1]

    # Sudden jump in translation set to zero
    step_limit = 1
    dd_row[abs(dd_row)>step_limit] = 0
    dd_col[abs(dd_col)>step_limit] = 0

    # Adjusted translation
    d_row[0] = 0
    d_col[0] = 0
    d_row[1:] = np.cumsum(dd_row)
    d_col[1:] = np.cumsum(dd_col)

    # Offset mid to zero
    drift_row = d_row - int(np.median(d_row))
    drift_col = d_col - int(np.median(d_col))

    # Translate images
    for i in range(len(I)):
        I[i,] = np.roll(I[i,], drift_row[i], axis=0)
        I[i,] = np.roll(I[i,], drift_col[i], axis=1)        
    
    return drift_row, drift_col, I


def reject_outliers(data, row, col):
    m = 3
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m], row[s<m], col[s<m]


def get_trace(I, row, col, spot_size):  
    r = row
    c = col
    s = int((spot_size-1)/2)
    return np.mean(np.mean(I[:,r-s:r+s+1,c-s:c+s+1], axis=2), axis=1)

def fit_trace(I_trace):
    n_frame = len(I_trace)
                
    # HMM      
    X = I_trace.reshape(n_frame, 1)
          
    # Set a new model for traidning
    param=set(X.ravel())
    remodel = hmm.GaussianHMM(n_components=2, covariance_type="full", 
        n_iter=100, params=param)        
        
    # Set initial parameters for training
    remodel.startprob_ = np.array([0.9, 0.1])
    remodel.transmat_ = np.array([[0.99, 0.01], 
                                  [0.1, 0.9]])
    remodel.means_ = np.array([X.min(), X.max()])
    remodel.covars_ = np.tile(np.identity(1), (2, 1, 1)) * (X.max()-X.min()) * 0.1        
           
    # Estimate model parameters (training)
    remodel.fit(X)

    # Find most likely state sequence corresponding to X
    Z = remodel.predict(X)
        
    # Reorder state number such that X[Z=0] < X[Z=1] 
    if X[Z==0].mean() > X[Z==1].mean():
        Z = 1 - Z
        remodel.transmat_ = np.array(
            [[remodel.transmat_[1][1], remodel.transmat_[1][0]],
             [remodel.transmat_[0][1], remodel.transmat_[0][0]]])
    
    # Transition probability
    tp_ub = remodel.transmat_[0][1] 
    tp_bu = remodel.transmat_[1][0]
        
    # Intensity trace fit      
    I_fit = np.zeros((n_frame))
    I_fit[Z==0] = X[Z==0].mean()  
    I_fit[Z==1] = X[Z==1].mean()                            

    return I_fit, tp_ub, tp_bu

   


def running_avg(x, n):
    return np.convolve(x, np.ones((n,))/n, mode='valid')  
   
def Exp(a, x):
    return np.exp(-(x)/a)/a * (0.5*(np.sign(x)+1))  
     
def exp1(x, a, b, c):
    return a + b * np.exp(-x/c)  

def exp2(x, a, b, c, d, e):
    return a + b * np.exp(-x/c) + d* np.exp(-x/e)  

# Exponential function with cutoff at x = b 
def Exp_cutoff(a, b, x):
    return (np.exp(-(x-b)/a)/a) * (0.5*(np.sign(x-b)+1)) + 1e-100

def Exp2_cutoff(a, b, c, d, x):
    return (c*(np.exp(-(x-d)/a)/a) + (1-c)*(np.exp(-(x-d)/b)/b)) * (0.5*(np.sign(x-d)+1)) + 1e-100   

# LogLikelihood 
def LL2(param, d, x):      
    [a, b, c] = param
    return np.sum(np.log10(Exp2_cutoff(a, b, c, d, x)))  

def MLE2(a, b, c, d, x): 
    fun = lambda *args: -LL2(*args)
    p0 = [a, b, c]
    result = minimize(fun, p0, method='SLSQP', args=(d, x)) 
    print(result)
    return result

def LL1(param, b, x):      
    [a] = param
    return np.sum(np.log10(Exp_cutoff(a, b, x)))  

def MLE1(a, b, x): 
    fun = lambda *args: -LL1(*args)
    p0 = [a]
    result = minimize(fun, p0, method='SLSQP', args=(b, x)) 
#    print(result)
    return result



def find_dwelltime(dwells):
    x = np.array(dwells)
    result1 = MLE1(np.mean(dwells), np.min(dwells), x)
    dwell_fit1 = result1["x"]
    return dwell_fit1
#       result2 = MLE2(np.mean(self.dwells)/2, np.mean(self.dwells)*2, 0.5, np.min(self.dwells), x)
#       self.dwell_fit2 = result2["x"]
  


def find_dwell(trace):
    """
    Find dwells of class1-4 return each list
    
    input: normalized trace of binary signal [0 or 1]
    output: list of each dwell class
    """
    
    tb = [] # Frame at binding
    tu = [] # Frame at unbinding
    
    # Find binding and unbinding moment
    for i in range(len(trace)-1):
        if trace[i] == 1 and trace[i+1] == 0: # unbinding
            tu.append(i) 
        elif trace[i] == 0 and trace[i+1] == 1: # binding
            tb.append(i) 
        else:
            continue

    # Dwell time of each class        
    t1 = [] # pre-existing binding
    t2 = [] # complete binding/unbinding
    t3 = [] # unfinished binding
    t4 = [] # pre-existing and unfinished binding

    # If there's no transition
    if len(tb) == 0 and len(tu) == 0:
        if trace[0] == 1: # All bound state
            t4 = [len(trace)]
        return t1, t2, t3, t4

    # Only one unbinding
    if len(tb) == 0 and len(tu) == 1:
        t1 = [tu[0]+1]
        return t1, t2, t3, t4

    # Only one binding
    if len(tb) == 1 and len(tu) == 0:
        t3 = [len(trace)-tb[0]-1]
        return t1, t2, t3, t4        
        
    # One binding and one unbinding
    if len(tb) == 1 and len(tu) == 1:        
        if tb[0] < tu[0]: # First binding, second unbinding
            t2 = [tu[0]-tb[0]]
        else: # First unbinding, second binding
            t1 = [tu[0]+1]            
            t3 = [len(trace)-tb[0]-1]
        return t1, t2, t3, t4           
        
    # All other cases below
    
    # Unbinding occurs before binding
    if tu[0] < tb[0]:
        t1 = [tu[0]+1]
        tu = tu[1:]
        
    # If last binding is unfinished
    if len(tb) > len(tu):
        t3 = [len(trace)-tb[-1]-1]
        tb = tb[:-1]         
        
    # Complete binding and unbinding
    t2 = [tu[i]-tb[i] for i in range(len(tb))]
        
    return t1, t2, t3, t4 
    
        



