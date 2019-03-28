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


def simulate_trace(t_window, t_b, t_u):
    """
    Simulate a time trace of binary signal with given parameters 
    
    input: total time window, mean bound time, mean unbound time
    return: a sequence of binary state (0 or 1)
    """
    
    tp_bu = 1/t_b
    tp_ub = 1/t_u
    
    # Determine the initial state based 
    signal = []
    init_state = int(random.random() > )
    signal.append(init_state) # Initial state
    
