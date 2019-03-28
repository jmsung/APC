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

# Add one
def add_one(x):
    return x+1

# Capital case
def capital_case(x):
    return x.capitalize()

