# -*- coding: utf-8 -*-
"""Column_Esser.py
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

#set_device('cpp_standalone')
b2.prefs.codegen.target = 'numpy'

b2.start_scope() #clear variables
start = time.time() #Running time
warnings.filterwarnings('ignore')
###############################################################################
########                   Parameter Definitions                        #######
#################### ###########################################################
tab1 = pd.read_csv('Esser_table1.csv', nrows = 68, delimiter=' ', index_col=False) #Define input table

dict1 = tab1.to_dict("list")

# x0 = []
# for i, r in tab1.iterrows():
#     x0.append(r.loc['Strength']) #Weights
    
#x = [2.00022501, 2.00014482, 2.00027791, 1.00007538, 1.00013961,
#       1.00046981, 1.00031536, 1.0003379 , 3.25040195, 3.25016564,
#       3.00028373, 2.99914853, 0.5003737 , 0.50051599]

##Simulation Parameters
duration = 1200*ms     # Total simulation time
sim_dt = 0.1*ms           # Integrator/sampling step

# #Set TMS
# TMS = False

# #Multiple model runs
# var_range = [0] #np.linspace(0, 0.25, 50)
# #output_rates = []
# L23E_output_rates = []
# L23I_output_rates = []
# L5E_output_rates = []
# L5I_output_rates = []
#for var in var_range:

    ##Longer timed TMS Simulation
#    a = 0
#    ta = b2.TimedArray(np.hstack((np.zeros(100), a, 0))*mV, dt = 3*ms)
#def objective(x):
###############################################################################
########                     Neuron Equations                           #######
###############################################################################
eqs = fl.equation('b2genn')

#eqs += 'I_syn = (v - Erev_AMPA)*g_AMPAa + (v - Erev_AMPA)*g_AMPAb + (v - Erev_AMPA)*g_AMPAc + (v - Erev_AMPA)*g_GABAAb + (v - Erev_AMPA)*g_GABAAc +' + '+'.join(['(v - Erev_{}) * g_{}{}'.format(r.loc['Transmitter'],r.loc['Transmitter'],i) for i,r in tab1.iterrows()]) + ' : volt\n'
#eqs += 'g_AMPAa : 1\n g_AMPAb : 1\n g_AMPAc : 1\n g_GABAAb : 1\n g_GABAAc : 1\n' + ''.join(['g_{}{} : 1\n'.format(r.loc['Transmitter'], i) for i,r in tab1.iterrows()])

###############################################################################
########                      Create Neurons                            #######
###############################################################################
num_cols = 25 #1, 2, 8, 32, 128  
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
TMS = b2.SpikeGeneratorGroup(1, [0], [300]*ms)

#     TMS = b2.SpikeGeneratorGroup(1, [0], [250]*ms)

# ###############################################################################
# ########                          Synapses                              #######
# ###############################################################################      
#Excitatory
TMS_synapse_0_E = b2.Synapses(TMS, columnsgroup_0, fl.equation('b2genn_synapse'), method = 'rk4', on_pre='x_{}_post += w'.format('AMPA'))
TMS_synapse_180_E = b2.Synapses(TMS, columnsgroup_180, fl.equation('b2genn_synapse'), method = 'rk4', on_pre='x_{}_post += w'.format('AMPA'))
TMS_synapse_0_E.connect(p=0.25)
TMS_synapse_180_E.connect(p=0.25)
TMS_synapse_0_E.w = 5
TMS_synapse_180_E.w = 5

# #Inhibitory
# TMS_synapse_0_I = b2.Synapses(TMS, columnsgroup_0, fl.equation('synapse').format(tr='GABAA',st = 'b'), method = 'rk4', on_pre='x_{}{} += w'.format('GABAA', 'b'))
# TMS_synapse_180_I = b2.Synapses(TMS, columnsgroup_180, fl.equation('synapse').format(tr='GABAA',st = 'c'), method = 'rk4', on_pre='x_{}{} += w'.format('GABAA', 'c'))
# TMS_synapse_0_I.connect(p=0.2)
# TMS_synapse_180_I.connect(p=0.2)
# TMS_synapse_0_I.w = 1
# TMS_synapse_180_I.w = 1

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
net.run(duration, profile = True) #Run
t14 = time.time()
print('Run Time:', t14 - t13)

print(b2.profiling_summary(net = net, show = 10))

# L23E_output_rates.append(spikemonL23E.num_spikes)
# L23I_output_rates.append(spikemonL23I.num_spikes)
# L5E_output_rates.append(spikemonL5E.num_spikes)
# L5I_output_rates.append(spikemonL5I.num_spikes)

###############################################################################
########                         Output                                 #######
###############################################################################

#fig, axes = plt.subplots(2,1, figsize=(10,8), sharex=True)
#axes[0].plot(var_range, L23E_output_rates, 'C1', label = 'L23E')
#axes[0].plot(var_range, L23I_output_rates, '-k', label = 'L23I')
#axes[1].plot(var_range, L5E_output_rates, 'C1', label = 'L5E')
#axes[1].plot(var_range, L5I_output_rates, '-k', label = 'L5I')

####Connectivity Plots5
#fl.visualise.neuron_connectivity(1,synapses_group,Neurons)
#fl.visualise.connectivity_distances(columnsgroup_0, all_synapses)

#Firing rate for each layer
# L23_firing = (spikemonL23E.num_spikes + spikemonL23I.num_spikes)/75*num_cols
# L5_firing = (spikemonL5E.num_spikes + spikemonL5I.num_spikes)/75*num_cols
# L6_firing = (spikemonL6E.num_spikes + spikemonL6I.num_spikes)/75*num_cols

L23_firing = (np.count_nonzero((spikemonL23E.t > 200*ms) * spikemonL23E.t))/(75*num_cols)
L5_firing = (np.count_nonzero((spikemonL5E.t > 200*ms) * spikemonL5E.t))/(75*num_cols)
L6_firing = (np.count_nonzero((spikemonL6E.t > 200*ms) * spikemonL6E.t))/(75*num_cols)
THA_firing = (np.count_nonzero((inputspikemon_TH.t > 200*ms) * inputspikemon_TH.t))/(75*num_cols)

print('L2/3 firing:', L23_firing)
print('L5 firing:', L5_firing)
print('L6 firing:', L6_firing)
print('Thalamus firing:', THA_firing)

# errors = (0.5 - L23_firing)**2 + (0.7 - L5_firing)**2 + (0.2 - L6_firing)**2
# print('error:', errors)

# return errors

# fprime = lambda x: scipy.optimize.approx_fprime(x, objective, 0.01)

# sol = minimize(objective, x0, method = 'Newton-CG', jac = fprime) # bounds=bnds
# print(sol)
t15 = time.time()
time0 = 0
index = [0, 0, 0, 0, 0]
##Only look at data after a certain time 
##timearrays = [spikemonL23.t, spikemonL5.t, spikemonL6.t, inputspikemon.t, spikemon.t]
##for j in range(len(timearrays)):
##    for i in range(len(timearrays[j])):
##        if timearrays[j][i] < time0*ms:
##            index[j] = i 
#
# #Plot Cortex Membrane Potentials
arraynum = time0*10
fig, ax = plt.subplots(5,1, figsize=(12,13), sharex=True)
plt.figure(figsize=(12, 5))
plt.subplot(2,1,1)
ax[0].plot(statemon.t[arraynum:]/ms, statemon.v[25*num_cols][arraynum:], 'C0', label='L3E')
ax[0].plot(statemon.t[arraynum:]/ms, statemon.v[57*num_cols][arraynum:], 'C1', label='L3I')
ax[0].plot(statemon.t[arraynum:]/ms, statemon.v[100*num_cols][arraynum:], 'C2', label='L5E')
ax[0].plot(statemon.t[arraynum:]/ms, statemon.v[142*num_cols][arraynum:], 'C3', label='L5I')
ax[0].plot(statemon.t[arraynum:]/ms, statemon.v[175*num_cols][arraynum:], 'C4', label='L6E')
ax[0].plot(statemon.t[arraynum:]/ms, statemon.v[215*num_cols][arraynum:], 'C5', label='L6I')
ax[0].set_ylabel('Membrame potential (v)')
ax[0].set_xlabel('Time (ms)')
ax[0].legend()

ax[1].plot(spikemon.t[arraynum:]/b2.ms, spikemon.i[arraynum:], '.k')
ax[1].set_ylabel('Neuron')

#### Plot Thalamus Membrane Potential ####
ax[2].plot(inputstatemon.t[arraynum:]/ms, inputstatemon.v[1][arraynum:], 'C6', label='MTE')
#plt.plot(inputstatemon.t[2000:]/ms, inputstatemon.v[120][2000:], 'C4', label='PME')
ax[2].set_ylabel('v')
ax[2].legend()
ax[3].plot(inputspikemon_TH.t[arraynum:]/ms, inputspikemon_TH.i[arraynum:], '.k')
ax[3].set_ylabel('Input Neurons (TH)')
ax[4].plot(inputspikemon_PS.t[arraynum:]/ms, inputspikemon_PS.i[arraynum:], '.k')
ax[4].set_ylabel('Input Neurons (PS)')                                                                                                      
t16 = time.time()
print('Plot Time:', t16-t15)
#####Extra Plots###
##L23 = sum(statemon.v[0:50*num_cols])/(50*num_cols)
##L5 = sum(statemon.v[75*num_cols:125*num_cols])/(50*num_cols)
##L6 = sum(statemon.v[150*num_cols:200*num_cols])/(50*num_cols)
##
##Iwave = (L23 + L5 + L6)/3
##plt.figure()
##plt.plot(Iwave)
#
##fig, ax = plt.subplots(6,1, figsize=(12,13), sharex=True)
##ax[0].plot(statemon.t[2400:3500]/ms, statemon.v[25*num_cols][2400:3500], 'C0', label='L3E')
##ax[1].plot(statemon.t[2400:3500]/ms, statemon.v[57*num_cols][2400:3500], 'C1', label='L3I')
##ax[2].plot(statemon.t[2400:3500]/ms, statemon.v[100*num_cols][2400:3500], 'C2', label='L5E')
##ax[3].plot(statemon.t[2400:3500]/ms, statemon.v[142*num_cols][2400:3500], 'C3', label='L5I')
##ax[4].plot(statemon.t[2400:3500]/ms, statemon.v[175*num_cols][2400:3500], 'C4', label='L6E')
##ax[5].plot(statemon.t[2400:3500]/ms, statemon.v[215*num_cols][2400:3500], 'C5', label='L6I')
##ax[2].set_ylabel('Membrame potential (v)')
##ax[5].set_xlabel('Time (ms)')
#
#fig, ax = plt.subplots(6,1, figsize=(12,13), sharex=True, sharey=True)
#ax[0].plot(statemon.t[:]/ms, statemon.v[25*num_cols][:], 'C2', label='L3E')
#ax[1].plot(statemon.t[:]/ms, statemon.v[57*num_cols][:], 'C3', label='L3I')
#ax[2].plot(statemon.t[:]/ms, statemon.v[100*num_cols][:], 'C2', label='L5E')
#ax[3].plot(statemon.t[:]/ms, statemon.v[142*num_cols][:], 'C3', label='L5I')
#ax[4].plot(statemon.t[:]/ms, statemon.v[175*num_cols][:], 'C2', label='L6E')
#ax[5].plot(statemon.t[:]/ms, statemon.v[215*num_cols][:], 'C3', label='L6I')
#plt.ylim(-0.08, 0.02)
#ax[2].set_ylabel('Membrame potential (v)')
#ax[5].set_xlabel('Time (ms)')
#
#plt.figure()
#plt.plot(inputstatemon.t[:]/ms, inputstatemon.v[53*num_cols][:], 'C5', label='L6I')

#######  Connectivity  ######
#src_indexes = []
#tgt_indexes = []
#
#for k in range(len(all_synapses)):
#    
#    
#    sources = all_synapses[k].i
#    targets = all_synapses[k].j
#    n_types = ['L2/3E', 'L2/3I', 'L5E', 'L5I', 'L6E', 'L6I', 'MTE', 'MTI', 'RI', 'SIE', 'PME']
#    
#    if src_group[k] == 'L2/3E':
#        sources = sources
#    if src_group[k] == 'L2/3I':
#        sources = sources + 49 * num_cols
#    if src_group[k] == 'L5E':
#        sources = sources + 74 * num_cols
#    if src_group[k] == 'L5I':
#        sources = sources + 124 * num_cols
#    if src_group[k] == 'L6E':
#        sources = sources + 149 * num_cols
#    if src_group[k] == 'L6I':
#        sources = sources + 199 * num_cols
#    if src_group[k] == 'MTE':
#        sources = sources + 224 * num_cols
#    if src_group[k] == 'MTI':
#        sources = sources + 236 * num_cols
#    if src_group[k] == 'RI':
#        sources = sources + 242 * num_cols
#    if src_group[k] == 'SIE':
#        sources = sources + 249 * num_cols
#    if src_group[k] == 'PME':
#        sources = sources + 260 * num_cols
#        
#    if tgt_group[k] == 'L2/3E':
#        targets = targets
#    if tgt_group[k] == 'L2/3I':
#        targets = targets + 49 * num_cols
#    if tgt_group[k] == 'L5E':
#        targets = targets + 74 * num_cols
#    if tgt_group[k] == 'L5I':
#        targets = targets + 124 * num_cols
#    if tgt_group[k] == 'L6E':
#        targets = targets + 149 * num_cols
#    if tgt_group[k] == 'L6I':
#        targets = targets + 199 * num_cols
#    if tgt_group[k] == 'MTE':
#        targets = targets + 224 * num_cols
#    if tgt_group[k] == 'MTI':
#        targets = targets + 236 * num_cols
#    if tgt_group[k] == 'RI':
#        targets = targets + 242 * num_cols
#    if tgt_group[k] == 'SIE':
#        targets = targets + 249 * num_cols
#    if tgt_group[k] == 'PME':
#        targets = targets + 260 * num_cols
#        
#    src_indexes.extend(sources)
#    tgt_indexes.extend(targets)
#
#plt.figure(figsize=(12, 12))
#plt.plot(src_indexes, tgt_indexes, 'k.')
#plt.fill_between([0,50],[50], facecolor='green', alpha=0.4)
#plt.fill_between([75,125],[125], facecolor='blue', alpha=0.4)
#plt.fill_between([150,200],[200], facecolor='red', alpha=0.4)

##### Histograms of average membrane potential ####
#a = []
#b = []
#for i in range(225):
#    a.append(np.mean([statemon.v[i][2000:]]))
#    
#for i in range(25):
#    b.append(np.mean([inputstatemon.v[i][2000:]]))
#
#plt.figure(figsize=(12,8))
#plt.subplot(2,2,1).set_title('L2/3E Membrane potential')
#plt.hist(a[0:50],bins = 30) #L2/3E
#plt.subplot(2,2,2).set_title('L5E Membrane potential')
#plt.hist(a[75:125],bins = 30) #L5E
#plt.subplot(2,2,3).set_title('L6E Membrane potential')
#plt.hist(a[150:200],bins = 30) #L6E
#plt.subplot(2,2,4).set_title('Excitatory Thalamus Membrane potential')
#plt.hist(b,bins = 30) #L6E

#### Histograms of average firing rate ####
#uniqueValues23E, occurCount23E = np.unique(spikemonL23E.i[index[0]:], return_counts=True)
#uniqueValues23I, occurCount23I = np.unique(spikemonL23I.i[index[0]:], return_counts=True)
#frequencies23E = occurCount23E/((duration/ms)/1000)
#plt.figure(figsize=(12,8))
#plt.subplot(2,2,1)
#plt.gca().set_title('L2/3E Average Firing')
#plt.hist(frequencies23E,bins = 30)
#
#uniqueValues5E, occurCount5E = np.unique(spikemonL5E.i[index[1]:], return_counts=True)
#uniqueValues5I, occurCount5I = np.unique(spikemonL5I.i[index[0]:], return_counts=True)
#frequencies5E = occurCount5E/((duration/ms)/1000)
## np.mean(frequencies5E)
#plt.subplot(2,2,2)
#plt.gca().set_title('L5E Average Firing')
#plt.hist(frequencies5E,bins = 30)
#
#uniqueValues6E, occurCount6E = np.unique(spikemonL6E.i[index[2]:], return_counts=True)
#uniqueValues6I, occurCount6I = np.unique(spikemonL6I.i[index[0]:], return_counts=True)
#frequencies6E = occurCount6E/((duration/ms)/1000)
#plt.subplot(2,2,3)
#plt.gca().set_title('L6E Average Firing')
#plt.hist(frequencies6E,bins = 30)

#total_firing = np.concatenate([spikemonL23E.t, spikemonL23I.t, spikemonL5E.t, spikemonL5I.t, spikemonL6E.t, spikemonL6I.t])

##Thalamus
#uniqueValues_input, occurCount_input = np.unique(inputspikemon_TH.i[index[3]:], return_counts=True)
#frequencies_input = (occurCount_input/((duration/ms)/1000))
#plt.subplot(2,2,4)
#plt.gca().set_title('Thalamus Average Firing')
#plt.xlabel('Frequency (Hz)')
#plt.ylabel('Number of neurons')
#plt.hist(frequencies_input,bins = 30, range = (0,20))
#
#plt.figure(figsize=(6,4))
#plt.hist(total_firing,bins = 1000)
#plt.xlabel('time')
#plt.ylabel('frequency')
#plt.savefig('Plot.png', transparent=True)

### Colourmaps ####
# top = cm.get_cmap('Oranges_r', 128)
# bottom = cm.get_cmap('Blues', 128)
# newcolours = np.vstack((top(np.linspace(0, 1, 128)),
#                       bottom(np.linspace(0, 1, 128))))
#colours = mpl.colors.ListedColormap(newcolours, name='OrangeBlue')
#
#rdgy = cm.get_cmap('jet', 256)
##
###Colourmap of Cortex
#data = statemon.v[0:50]
#fig, axs = plt.subplots(figsize=(10, 4), constrained_layout=True)
#psm = axs.pcolormesh(data, cmap=rdgy, rasterized=True, vmin=-0.07, vmax=-0.05)
#fig.colorbar(psm, ax=axs)
#plt.show()
#
##Colourmap of Input Areas
#data = inputstatemon.v
#fig, axs = plt.subplots(figsize=(5, 2), constrained_layout=True)
#psm = axs.pcolormesh(data, cmap=viridis, rasterized=True, vmin=-0.07, vmax=-0.05)
#fig.colorbar(psm, ax=axs)
#plt.show()
# colours = mpl.colors.ListedColormap(newcolours, name='OrangeBlue')

# rdgy = cm.get_cmap('jet', 256)
# #
# ##Colourmap of Cortex
# data = statemon.v[0:50*num_cols]
# fig, axs = plt.subplots(figsize=(10, 4), constrained_layout=True)
# psm = axs.pcolormesh(data, cmap=rdgy, rasterized=True, vmin=-0.07, vmax=-0.05)
# fig.colorbar(psm, ax=axs)
# plt.show()

# #Colourmap of Input Areas
# data = inputstatemon.v
# fig, axs = plt.subplots(figsize=(5, 2), constrained_layout=True)
# psm = axs.pcolormesh(data, cmap=viridis, rasterized=True, vmin=-0.07, vmax=-0.05)
# fig.colorbar(psm, ax=axs)
# plt.show()

# ###### 3D spatial plot ####
# fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(columnsgroup_0.X/b2.umetre, columnsgroup_0.Y/b2.umetre, columnsgroup_0.Z/b2.umetre, marker='o')
# ax.scatter(columnsgroup_180.X/b2.umetre, columnsgroup_180.Y/b2.umetre, columnsgroup_0.Z/b2.umetre, marker='x')
# plt.xlabel("Distance (um)")
# plt.ylabel("Distance (um)")

####3D connectivity plot ###
#fl.visualise.spatial_connectivity(all_synapses, columnsgroup_0, source_idx, target_idx)

end = time.time()
print('Total Time Taken:', end-start)

# ######## I-wave generation ###########

# counter = collections.Counter(spikemonL5E.t[:]/b2.ms)
# key_values = {key:value for key, value in counter.items() if key >= 250.0 
#                   and key <= 260.0}
# frequencies = [ v for v in key_values.values() ]
# all_frequencies = [ v for v in counter.values() ]

# # a=[0,1]
# x=np.convolve(L5,frequencies)
# plt.figure()
# plt.plot(x)

###############################################################################
########                       Unused Code                              #######
###############################################################################
## Print V
#Check initial v
#print('Before v = %s' % column1.v[0])
##See what the final v is
#print('After v= %s' % column1.v[0])

##Defining Input Areas synapses
#    if (tgt == 'MTE') or (tgt == 'MTI') or (tgt == 'RI') or (tgt == 'SIE') or (tgt == 'PME'):
#\or(src == 'MTE') or (src == 'MTI') or (src == 'RI') or (src == 'SIE') or (src == 'PME'):
        #('MT' or 'R' or 'SI' or 'PM') in (src or tgt):
#        syn = b2.Synapses(neuron_group[src],
#                     neuron_group[tgt],
#                      model = common_model.format(i),
#                      on_pre='g_syn += w')
        #'v_post += w*volt'
#    else: 

##Using dict structure for synapses
#all_synapses = {}
     #all_synapses['{}2{}'.format(src, tgt)] = syn
#    print('{}2{}'.format(src, tgt))
#for k in all_synapses.keys():
#    varname = re.sub('[!@#$/]', '', k)
#    exec_str = 'syn_{:s} = all_synapses["{:s}"]'.format(varname, k)
#    print(exec_str)
#    exec(exec_str)

##Equations
##t_peak = ((tau_2*tau_1)/(tau_2 - tau_1)) * log (tau_2/tau_1)  : second

#n_syn = len(tab1)
#eqs += 'g = ' + ' + '.join(['g{:d}'.format(i) for i in range(n_syn)]) + ' : 1\n'
#eqs += ''.join(['g{:d} : 1\n'.format(i) for i in range(n_syn)])

#Moved section of synapse common model to the main eqs
#common_model = '''
#dg_syn/dt = ((tau_2 / tau_1) ** (tau_1 / (tau_2 - tau_1))*x-g_syn)/tau_1 : 1
#dx/dt = (-x/tau_2)                                              : 1
#g{:d}_post = g_syn : 1 (summed) 
#w : 1
#'''
 
#Removed from Main Eqs    
#g :1
#I_syn = (v - Erev_AMPA) * g_AMPA + (v - Erev_GABAA) * g_GABAA + (v - Erev_GABAB) * g_GABAB + (v - Erev_NMDA) * g_NMDA : volt

##Add SpikeGenerator
#indices = b2.array([0])
#times = b2.array([50])*ms
#G = b2.SpikeGeneratorGroup(1, indices, times)
#S = Synapses(G, neuron_group['L2/3'], eqs_syn, on_pre = 'g_syn += w')
#S.connect(p = 1)
#S.w = 0.5

## Connect spike generator
#tgts = ['L6', 'L5', 'L2/3']
#for i, r in enumerate(tgts):
##    print('*{:d}'.format(i))
#    gensyn = b2.Synapses(G, neuron_group[r],
#                         model=common_model.format(i + 18),
#                         on_pre='g_syn += w')
#    gensyn.connect(p = 1)
#    gensyn.w = 0.3
#    gen_synapses['G2{}'.format(r)] = gensyn

##Channel Parameters
#  #Synaptic current gPeak values
#gpeak_AMPA = 0.1
#gpeak_NMDA = 0.1
#gpeak_GABAA = 0.33
#gpeak_GABAB = 0.0132
#
#  #Synaptic current reversal potentials
#Erev_AMPA = 0*mV
#Erev_NMDA = 0*mV
#Erev_GABAA = -70*mV
#Erev_GABAB = -90*mV
#
#  #Synaptic current time constants
#tau1_AMPA = 0.5*ms
#tau2_AMPA = 2.4*ms
#tau1_NMDA = 4*ms
#tau2_NMDA = 40*ms
#tau1_GABAA = 1*ms
#tau2_GABAA = 7*ms
#tau1_GABAB = 60*ms
#tau2_GABAB = 200*ms
     
#plt.plot(statemon.t[0:500]/ms, statemon.v[175*num_cols][1500:2000], 'k', label='L5E')
#plt.xlabel('Time')
#plt.ylabel('Membrane Potential')
#plt.ylim(-0.08,-0.05
