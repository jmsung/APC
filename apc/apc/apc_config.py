# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:30:05 2019

@author: Jongmin Sung


"""

# config.py

import os
from pathlib import Path  
from inspect import currentframe, getframeinfo

fname = getframeinfo(currentframe()).filename # current file name
current_dir = Path(fname).resolve().parent
data_dir = Path(fname).resolve().parent.parent.parent/'data'  



