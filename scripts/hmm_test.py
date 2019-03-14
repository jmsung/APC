# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 19:56:50 2019 @author: Jongmin Sung

The code was modified from the following. 
https://hmmlearn.readthedocs.io/en/stable/index.html

Model: 
    Number of states = 2 [unbound, bound]
    Transition probability = [[tp_uu, tp_ub], [tp_bu, tp_bb]]
    Emission probability = [ep_u, ep_b]

"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import random

# User parameters
n_sample = 100
n_frame = 1000
SNR = 10
time_bound = 10
time_unbound = 50

# Start probability 
startprob = np.array([0.5, 0.5])

# The transition probability matrix
tp_ub = 1/time_unbound
tp_uu = 1 - tp_ub
tp_bu = 1/time_bound
tp_bb = 1 - tp_bu
transmat = np.array([[tp_uu, tp_ub],
                     [tp_bu, tp_bb]])

# The means of each state
means = np.array([[0.0],
                  [1.0]])

# The covariance of each component
covars = np.tile(np.identity(1), (2, 1, 1)) / SNR

# Build an HMM instance and set parameters
model = hmm.GaussianHMM(n_components=2, covariance_type="full")

# Set the parameters to generate samples
model.startprob_ = startprob
model.transmat_ = transmat
model.means_ = means
model.covars_ = covars


# Generate samples
X, Z_true = model.sample(n_frame)

X_true = np.zeros(n_frame)
for i in range(2):
    X_true[Z_true==i] = X[Z_true==i].mean()

# Set a new model for traidning
remodel = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)

# Set initial parameters for training
remodel.startprob_ = np.array([0.5, 0.5])
remodel.transmat_ = np.array([[0.5, 0.5], 
                              [0.5, 0.5]])
remodel.means_ = np.array([X.min(), X.max()])
remodel.covars_ = np.tile(np.identity(1), (2, 1, 1)) * random.uniform(0.1, 1)

# Estimate model parameters (training)
remodel.fit(X)  

# Find most likely state sequence corresponding to X
Z_predict = remodel.predict(X)

# Reorder state number such that X[Z=0] < X[Z=1] 
if X[Z_predict == 0].mean() > X[Z_predict == 1].mean():
    Z_predict = 1 - Z_predict

# Percent of error
percent_error = sum(abs(Z_true - Z_predict))/n_frame*100
print("Prediction error = ", percent_error, "%")

# Sequence of predicted states
X_predict = np.zeros(n_frame)
for i in range(2):
    X_predict[Z_predict == i] = X[Z_predict == i].mean()

# Plot the sampled data
plt.figure(1, clear=True)
plt.plot(X, "k", label="observations", ms=1, lw=1, alpha=0.5)
plt.plot(X_true, "b", label="states", lw=5, alpha=1)
plt.plot(X_predict, "r", label="predictions", lw=5, alpha=0.7)
plt.title("SNR = %.1f, Error = %.1f %%" % (SNR, percent_error))

plt.show()