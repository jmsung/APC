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


def generate_trace(t_total, t_b, t_u):
    """
    Generate a time trace of binary signal with given parameters 
    
    input: total time window, mean bound time, mean unbound time
    output: array of binary state (0 or 1)
    """
    
    # Transition probability. This could be either tp = 1/t or tp = 1/(t+1)
    tp_bu = 1-np.exp(-1/t_b)
    tp_ub = 1-np.exp(-1/t_u)
    
    # Initialize that all the states are unbound
    signal = np.zeros(t_total) 

    for i in range(t_total-1):
        if signal[i] == 0: # current state is unbound
            if random.random() < tp_ub: # transition toward bound
                signal[i+1] = 1
        else: # current state is bound
            if random.random() > tp_bu: # stay in bound
                signal[i+1] = 1       
                
    return signal
    

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
    
        



