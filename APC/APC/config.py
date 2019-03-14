# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:30:05 2019

@author: Jongmin Sung


In projectname/projectname/config.py, we place in special paths and variables 
that are used across the project. An example might be:


"""

# config.py

import os
from pathlib import Path  # pathlib is seriously awesome!
from inspect import currentframe, getframeinfo


fname = getframeinfo(currentframe()).filename # current file name

data_dir = Path(fname).resolve().parent.parent.parent / 'data' / 'raw'
data_files = os.listdir(data_dir)


print(data_dir)
print(data_files)
