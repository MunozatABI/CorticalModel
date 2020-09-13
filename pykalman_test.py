# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 09:17:50 2020

@author: lmun373
"""
import numpy as np
import brian2 as b2

from pykalman import UnscentedKalmanFilter

#Governing Equations
#x_0 ~ Normal(nu, sigma)
#x_t+1 = f(x_t, Normal(0,Q))
#z_t = g(x_t, Normal(0,R))

#Transition Function, produces estimated state at time t+1
def f(x, w):
    return x + np.sin(w)

#Observation Function, produces observation at time t
def g(x, v):
    return x + v

def model(weight):
    b2.start_scope()
    # Parameters
    num_inputs = 100
    input_rate = 10*b2.Hz
    tau = 1*b2.ms
    # Use this list to store output rates
    output_rates = []
    P = b2.PoissonGroup(num_inputs, rates=input_rate)
    eqs = '''
    dv/dt = -v/tau : 1
    w : 1
    '''
    G = b2.NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='exact')
    S = b2.Synapses(P, G, on_pre='v += w')
    S.connect()
    S.w = weight
    M = b2.SpikeMonitor(G)
    # Run it and store the output firing rate in the list
    b2.run(1*b2.second)
    output_rates.append(M.num_spikes/b2.second)
    return output_rates

def observation(weight, goal):
    output = model(weight)
    error = output[0]/b2.Hz - goal
    return error ### This has to return a value that I want to match with the model output

#Define UKF
ukf = UnscentedKalmanFilter(f, g, observation_covariance=0.1)
#ukf = UnscentedKalmanFilter(model, observation, observation_covariance=0.1)

#Run UKF, with Z(t) observations and return the means and covariances of state distributions over time t
(filtered_state_means, filtered_state_covariances) = ukf.filter([0, 1, 2]) #Returns means - values of x which match f to g; and covariances variability
(smoothed_state_means, smoothed_state_covariances) = ukf.smooth([0, 1, 2]) # Applies smoother
#(filtered_state_means, filtered_state_covariances) = ukf.filter([100])