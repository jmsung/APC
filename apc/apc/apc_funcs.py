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
#import pandas as pd

def find_movie(directory):
    for file in os.listdir(directory):
        if file.endswith(".tif"):
            filename = file
    return filename

def read_movie1(movie_path):
    movie = Image.open(movie_path)
    n_frame = movie.n_frames
    n_row = movie.size[1]
    n_col = movie.size[0]
    print('[frame, row, col] = [',n_frame, n_row, n_col,']')        
          
    # Read tif file and save into I[frame, row, col]
    I = np.zeros((n_frame, n_row, n_col), dtype=int)
    for i in range(n_frame): 
        movie.seek(i) # Move to i-th frame
        I[i,] = np.array(movie, dtype=int)    
    return I

def read_movie2(movie_path):
    # read tiff file
    with TiffFile(movie_path) as tif:
        imagej_hyperstack = tif.asarray()
        imagej_metadata = tif.imagej_metadata

    f = open('meta_data.txt', 'w')
    f.write(str(imagej_metadata))
    f.close()

    return imagej_hyperstack, imagej_metadata

def read_movie(movie_path):
    # read tiff file
    with TiffFile(movie_path) as tif:
        imagej_hyperstack = tif.asarray()
        imagej_metadata = str(tif.imagej_metadata)
        imagej_metadata = imagej_metadata.split(',')

#    for item in imagej_metadata:
#        print(item)

    # write meta_data    
    with open(movie_path.parent/'meta_data.txt', 'w') as f:
        for item in imagej_metadata:
            f.write(item+'\n')

    return imagej_hyperstack, imagej_metadata



def reject_outliers(data, m=3):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

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

            
def crop(self, I, x, y, s):
    hs = int(s/2)
    I0 = I[x-hs:x+hs+1, y-hs:y+hs+1]
    val = skimage.filters.threshold_otsu(I0)
    mask = I0 > val
    return mask
        
                      
def drift(self):
    r = 20  
    size = min(self.n_row, self.n_row) - (2*r+10)
    I0 = self.I[0,] # 0th frame 
    cx = int(self.n_row/2)
    cy = int(self.n_col/2) 
    I0s = self.crop(I0, cx, cy, size)     
    self.I0s = I0s
     
    self.drift = np.zeros((self.n_frame, 2*r+1, 2*r+1), dtype=float)  
     
    self.drift_x = []
    self.drift_y = []

    for i in range(self.n_frame):  
        print(i)
        I1 = self.I[i,] # ith frame  
        for j in range(-r, r+1):
            for k in range(-r, r+1):
                I1s = self.crop(I1, cx+j, cy+k, size)
                corr = np.sum(I0s*I1s)
                self.drift[i, j+r, k+r] = corr
                                    
        self.drift[i,] = self.drift[i,] - self.drift[i,].min()
        self.drift[i,] = self.drift[i,]/self.drift[i,].max()
        dr = np.argwhere(self.drift[i,] == 1)
        self.drift_x += [dr[0][1]-r]
        self.drift_y += [dr[0][0]-r]
        
def drift_correct(self):
    I = np.zeros((self.n_frame, self.n_row, self.n_col), dtype=int)
    dx = self.drift_x
    dy = self.drift_y
        
    for i in range(self.n_frame):  
        for j in range(self.n_row):
            if j-dy[i] >= 0 and j-dy[i] < self.n_row: 
                for k in range(self.n_col):
                    if k-dx[i] >= 0 and k-dx[i] < self.n_col:                    
                        I[i, j-dy[i], k-dx[i]] = self.I[i, j, k]                                  
    self.I = I
                   
def offset(self):
    I_min = np.min(self.I, axis=0)
    for i in range(self.n_frame):
        self.I[i,] = self.I[i,] - I_min

# Find local maxima from movie.I_max                    
def find_peaks(self):#, spot_size): 
    I = self.I_max
#       self.peaks = skimage.feature.peak_local_max(I, min_distance=int(spot_size*1.5))
    self.peaks = peak_local_max(I, min_distance=int(spot_size*1.5))        
    self.n_peaks = len(self.peaks[:, 1])
    print('\nFound', self.n_peaks, 'peaks. ')
        
# Find real molecules from the peaks
def find_mols(self):#, spot_size, SNR_min):#, dwell_min, dwell_max): 
    row = self.peaks[::-1,0]
    col = self.peaks[::-1,1]
    self.mols = []
    self.dwells = []
    self.noise = []
    self.SNR = []
    self.tp_ub = []
    self.tp_bu = []
    for i in range(self.n_peaks):
        mol = Mol(self.I, row[i], col[i])#, spot_size)
        mol.normalize()
        mol.find_noise()
#           if mol.evaluate(SNR_min, dwell_min, dwell_max) is True:
        if mol.evaluate() is True:
            self.mols.append(mol)    
            self.dwells.extend(mol.dwell)
            self.noise.append(mol.noise)
            self.SNR.extend(mol.SNR)
            self.tp_ub.append(mol.tp_ub)
            self.tp_bu.append(mol.tp_bu)            
    print('Found', len(self.mols), 'molecules. \n')  
 
def find_dwelltime(dwells):
    x = np.array(dwells)
    result1 = MLE1(np.mean(dwells), np.min(dwells), x)
    dwell_fit1 = result1["x"]
    return dwell_fit1
#       result2 = MLE2(np.mean(self.dwells)/2, np.mean(self.dwells)*2, 0.5, np.min(self.dwells), x)
#       self.dwell_fit2 = result2["x"]
  

def drift_correct(I_ref, I_frame):
    """
    Input: frames of 2D image
    Return: translation vector 
    """

    drift = []

    for ix in range(len(I_frame)):
        result = translation(I_ref, I_frame[ix])
 
        drift.append(result['tvec'])
    
    return drift


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
    
        



