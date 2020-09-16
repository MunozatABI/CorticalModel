# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:49:38 2020

@author: lmun373
"""
###############################################################################
########                   Import Libraries                             #######
###############################################################################

import brian2 as b2
from brian2 import mV, ms, set_device
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D 
import scipy
from scipy.optimize import minimize
import pandas as pd
import function_library as fl
import numpy as np
import time
import parameters
from parameters import *
import warnings
import collections
import brian2genn

set_device('genn')

b2.start_scope() #clear variables
start = time.time() #Running time
warnings.filterwarnings('ignore')
###############################################################################
########                   Parameter Definitions                        #######
#################### ###########################################################
tab1 = pd.read_csv('Esser_table2.csv', nrows = 68, delimiter=' ', index_col=False) #Define input table

##Simulation Parameters
duration = 1200*ms     # Total simulation time
sim_dt = 0.1*ms           # Integrator/sampling step

# #Set TMS
# TMS = False

    ##Longer timed TMS Simulation
#    a = 0
#    ta = b2.TimedArray(np.hstack((np.zeros(100), a, 0))*mV, dt = 3*ms)
#def objective(x):
###############################################################################
########                     Neuron Equations                           #######
###############################################################################
eqs = fl.equation('b2genn')

###############################################################################
########                      Create Neurons                            #######
###############################################################################
num_cols = 1 #1, 2, 8, 32, 128
columnsgroup_0 = []
t1 = time.time()
columnsgroup_0 = fl.generate.column(num_cols,eqs,0)
t2 = time.time()
print('Time for column generation:', t2-t1)
columnsgroup_180 = fl.generate.column(num_cols,eqs,180)

#Input Areas - SMA, PME, THA, RN
in_type = ['THA']
in_num = [75*num_cols]
t3 = time.time()
Input_Neurons = fl.generate.neurons(in_num, in_type, eqs, 1, 0)
t4 = time.time()
print('Time for input neuron generation:', t4-t3)

T_R = int((in_num[0]/3) * 2)
PM_SI = int(in_num[0]/3)

t5 = time.time()
Spike = fl.generate.poissonspikes(T_R, PM_SI, duration)
t6 = time.time()
print('Time for spike generation:', t6-t5)

neuron_group = {'L2/3E0': columnsgroup_0[0:50*num_cols],
                'L2/3I0': columnsgroup_0[50*num_cols:75*num_cols], 
                'L5E0': columnsgroup_0[75*num_cols:125*num_cols], 
                'L5I0': columnsgroup_0[125*num_cols:150*num_cols], 
                'L6E0': columnsgroup_0[150*num_cols:200*num_cols], 
                'L6I0': columnsgroup_0[200*num_cols:225*num_cols],
                'L2/3E180': columnsgroup_180[0:50*num_cols],
                'L2/3I180': columnsgroup_180[50*num_cols:75*num_cols], 
                'L5E180': columnsgroup_180[75*num_cols:125*num_cols], 
                'L5I180': columnsgroup_180[125*num_cols:150*num_cols], 
                'L6E180': columnsgroup_180[150*num_cols:200*num_cols], 
                'L6I180': columnsgroup_180[200*num_cols:225*num_cols],
                'MTE': Input_Neurons[0:24*num_cols],
                'MTI': Input_Neurons[24*num_cols:36*num_cols],
                'RI': Input_Neurons[36*num_cols:50*num_cols],
                'THA':Input_Neurons[0:50*num_cols],
                'SIE': Input_Neurons[50*num_cols:62*num_cols],
                'PME': Input_Neurons[62*num_cols:75*num_cols], 
                'PMSI': Input_Neurons[50*num_cols:75*num_cols]
                 }

#Model of TMS activation
# if TMS == True:
#     b2.SpikeGeneratorGroup(1, [0], [250]*ms)

#     TMS = b2.SpikeGeneratorGroup(1, [0], [250]*ms)

# ###############################################################################
# ########                          Synapses                              #######
# ###############################################################################      

t7 = time.time()
Input_synapses = fl.generate.synapses([Spike], [Input_Neurons], ['AMPA'], [1], [1], [0])
t8 = time.time()
print('Time for input synapse generation:', t8-t7)


t9 = time.time()
src_group, tgt_group, all_synapses = fl.generate.model_synapses(tab1, neuron_group)
#src_group_dict, tgt_group_dict, all_synapses_dict = fl.generate.model_dict_synapses(dict1, neuron_group)
t10 = time.time()
print('Time for synapse generation:', t10-t9)
#source_idx, target_idx = fl.subgroup_idx(src_group, tgt_group)

###############################################################################
########                         Monitors                               #######
###############################################################################
t11 = time.time()
statemon = b2.StateMonitor(columnsgroup_0, 'v', record=range(225*num_cols))
thetamon = b2.StateMonitor(columnsgroup_0, 'theta', record=range(225*num_cols))
spikemon = b2.SpikeMonitor(columnsgroup_0, variables = ['v', 't'])
spikemon_generator = b2.SpikeMonitor(Spike, variables = ['t'])
spikemonL23E = b2.SpikeMonitor(neuron_group['L2/3E0'], variables = ['v', 't'])
spikemonL5E = b2.SpikeMonitor(neuron_group['L5E0'], variables = ['v', 't'])
spikemonL6E = b2.SpikeMonitor(neuron_group['L6E0'], variables = ['v', 't'])
spikemonL23I = b2.SpikeMonitor(neuron_group['L2/3I0'], variables = ['v', 't'])
spikemonL5I = b2.SpikeMonitor(neuron_group['L5I0'], variables = ['v', 't'])
spikemonL6I = b2.SpikeMonitor(neuron_group['L6I0'], variables = ['v', 't'])
inputstatemon = b2.StateMonitor(Input_Neurons, 'v', record=range(75*num_cols))
inputspikemon_TH = b2.SpikeMonitor(neuron_group['MTE'], variables = ['v', 't'])
inputspikemon_PS = b2.SpikeMonitor(neuron_group['PMSI'], variables = ['v', 't'])
t12 = time.time()
print('Time for spikemonitor definition:', t12-t11)
###############################################################################
########                         Run Model                              #######
###############################################################################
net = b2.Network(b2.collect())  #Automatically add visible objects 
net.add(Input_synapses, all_synapses)           #Manually add list of synapses # TMS_synapse_0, TMS_synapse_180

t13 = time.time()
net.run(duration) #Run
t14 = time.time()
print('Run Time:', t14 - t13)

###############################################################################
########                         Output                                 #######
###############################################################################

#Firing rate for each layer
# L23_firing = (spikemonL23E.num_spikes + spikemonL23I.num_spikes)/75*num_cols
# L5_firing = (spikemonL5E.num_spikes + spikemonL5I.num_spikes)/75*num_cols
# L6_firing = (spikemonL6E.num_spikes + spikemonL6I.num_spikes)/75*num_cols

L23_firing = (np.count_nonzero((spikemonL23E.t > 200*ms) * spikemonL23E.t))/75*num_cols
L5_firing = (np.count_nonzero((spikemonL5E.t > 200*ms) * spikemonL5E.t))/75*num_cols
L6_firing = (np.count_nonzero((spikemonL6E.t > 200*ms) * spikemonL6E.t))/75*num_cols
THA_firing = (np.count_nonzero((inputspikemon_TH.t > 200*ms) * inputspikemon_TH.t))/75*num_cols

print('L2/3 firing:', L23_firing)
print('L5 firing:', L5_firing)
print('L6 firing:', L6_firing)
print('Thalamus firing:', THA_firing)
