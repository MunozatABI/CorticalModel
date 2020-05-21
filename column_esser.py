5# -*- coding: utf-8 -*-
"""Column_Esser.ipynb
In this script, I am aiming to make a column of 225 Neurons
"""
###############################################################################
########                   Import Libraries                             #######
###############################################################################

import brian2 as b2
from brian2 import mV, ms
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D 

import pandas as pd
import function_library as fl
import numpy as np
import time
import parameters
from parameters import *
import warnings
import collections

b2.start_scope() #clear variables
start = time.time() #Running time
warnings.filterwarnings('ignore')
###############################################################################
########                   Parameter Definitions                        #######
###############################################################################
tab1 = pd.read_csv('Esser_table2.csv', nrows = 2, delimiter=' ', index_col=False) #Define input table

##Simulation Parameters
duration = 1000*ms     # Total simulation time
sim_dt = 0.001*ms           # Integrator/sampling step

###############################################################################
########                     Neuron Equations                           #######
###############################################################################
eqs = fl.equation('current')
eqs += 'I_syn = (v - Erev_AMPA)*g_AMPAa + (v - Erev_AMPA)*g_AMPAb + (v - Erev_AMPA)*g_AMPAc +' + '+'.join(['(v - Erev_{}) * g_{}{}'.format(r.loc['Transmitter'],r.loc['Transmitter'],i) for i,r in tab1.iterrows()]) + ' : volt\n'
eqs += 'g_AMPAa : 1\n g_AMPAb : 1\n g_AMPAc : 1\n' + ''.join(['g_{}{} : 1\n'.format(r.loc['Transmitter'], i) for i,r in tab1.iterrows()])

###############################################################################
########                      Create Neurons                            #######
###############################################################################
num_cols = 8 #2, 8, 32, 64
columnsgroup_0 = []
columnsgroup_0 = fl.generate.column(num_cols,eqs,0)
columnsgroup_180 = fl.generate.column(num_cols,eqs,180)

#Input Areas - SMA, PME, THA, RN
in_type = ['THA']
in_num = [75*num_cols]
Input_Neurons = fl.generate.neurons(in_num, in_type, eqs, 1, 0)

T_R = int((in_num[0]/3) * 2)
PM_SI = int(in_num[0]/3)

Spike = fl.generate.spikes(T_R, PM_SI, duration)

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
#
# TMS = b2.SpikeGeneratorGroup(1, [0], [250]*ms)
###############################################################################
########                          Synapses                              #######
###############################################################################
Input_synapses = fl.generate.synapses([Spike], [Input_Neurons], ['AMPA'], [1], [1], [0])

src_group, tgt_group, all_synapses = fl.generate.model_synapses(tab1, neuron_group)

# #Model of TMS activation
# TMS_synapse_0 = b2.Synapses(TMS, columnsgroup_0, fl.equation('synapse').format(tr='AMPA',st = 'b'), method = 'rk4', on_pre='x_{}{} += w'.format('AMPA', 'b'))
# TMS_synapse_180 = b2.Synapses(TMS, columnsgroup_180, fl.equation('synapse').format(tr='AMPA',st = 'c'), method = 'rk4', on_pre='x_{}{} += w'.format('AMPA', 'c'))
# TMS_synapse_0.connect(p=0.5)
# TMS_synapse_180.connect(p=0.5)
# TMS_synapse_0.w = 1
# TMS_synapse_180.w = 1

###############################################################################
########                         Monitors                               #######
###############################################################################
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

###############################################################################
########                         Run Model                              #######
###############################################################################
net = b2.Network(b2.collect())  #Automatically add visible objects 
net.add(Input_synapses, all_synapses)           #Manually add list of synapses # TMS_synapse_0, TMS_synapse_180

net.run(duration) #Run

###############################################################################
########                       Plot Graphs                              #######
###############################################################################
time0 = 0
index = [0, 0, 0, 0, 0]
#Only look at data after a certain time 
#timearrays = [spikemonL23.t, spikemonL5.t, spikemonL6.t, inputspikemon.t, spikemon.t]
#for j in range(len(timearrays)):
#    for i in range(len(timearrays[j])):
#        if timearrays[j][i] < time0*ms:
#            index[j] = i 

#Plot Cortex Membrane Potentials
arraynum = time0*10
fig, ax = plt.subplots(5,1, figsize=(12,13), sharex=True)
#plt.figure(figsize=(12, 5))
#plt.subplot(2,1,1)
ax[0].plot(statemon.t[arraynum:]/ms, statemon.v[25*num_cols][arraynum:], 'C0', label='L3E')
ax[0].plot(statemon.t[arraynum:]/ms, statemon.v[57*num_cols][arraynum:], 'C1', label='L3I')
ax[0].plot(statemon.t[arraynum:]/ms, statemon.v[100*num_cols][arraynum:], 'C2', label='L5E')
ax[0].plot(statemon.t[arraynum:]/ms, statemon.v[142*num_cols][arraynum:], 'C3', label='L5I')
ax[0].plot(statemon.t[arraynum:]/ms, statemon.v[175*num_cols][arraynum:], 'C4', label='L6E')
ax[0].plot(statemon.t[arraynum:]/ms, statemon.v[215*num_cols][arraynum:], 'C5', label='L6I')
ax[0].set_ylabel('Membrame potential (v)')
ax[0].set_xlabel('Time (ms)')
ax[0].legend()
ax[1].plot(spikemon.t[index[4]:]/b2.ms, spikemon.i[index[4]:], '.k')
ax[1].set_ylabel('Neuron')

#### Plot Thalamus Membrane Potential ####
ax[2].plot(inputstatemon.t[arraynum:]/ms, inputstatemon.v[1][arraynum:], 'C6', label='MTE')
#plt.plot(inputstatemon.t[2000:]/ms, inputstatemon.v[120][2000:], 'C4', label='PME')
ax[2].set_ylabel('v')
ax[2].legend()
ax[3].plot(inputspikemon_TH.t[index[3]:]/ms, inputspikemon_TH.i[index[3]:], '.k')
ax[3].set_ylabel('Neuron')
ax[4].plot(inputspikemon_PS.t[index[4]:]/ms, inputspikemon_PS.i[index[4]:], '.k')
ax[4].set_ylabel('v')

###Extra Plots###
L23 = sum(statemon.v[0:50*num_cols])/(50*num_cols)
L5 = sum(statemon.v[75*num_cols:125*num_cols])/(50*num_cols)
L6 = sum(statemon.v[150*num_cols:200*num_cols])/(50*num_cols)

Iwave = (L23 + L5 + L6)/3
plt.figure()
plt.plot(Iwave)

# fig, ax = plt.subplots(6,1, figsize=(12,13), sharex=True)
# ax[0].plot(statemon.t[2400:3500]/ms, statemon.v[25*num_cols][2400:3500], 'C0', label='L3E')
# ax[1].plot(statemon.t[2400:3500]/ms, statemon.v[57*num_cols][2400:3500], 'C1', label='L3I')
# ax[2].plot(statemon.t[2400:3500]/ms, statemon.v[100*num_cols][2400:3500], 'C2', label='L5E')
# ax[3].plot(statemon.t[2400:3500]/ms, statemon.v[142*num_cols][2400:3500], 'C3', label='L5I')
# ax[4].plot(statemon.t[2400:3500]/ms, statemon.v[175*num_cols][2400:3500], 'C4', label='L6E')
# ax[5].plot(statemon.t[2400:3500]/ms, statemon.v[215*num_cols][2400:3500], 'C5', label='L6I')
# ax[2].set_ylabel('Membrame potential (v)')
# ax[5].set_xlabel('Time (ms)')

fig, ax = plt.subplots(6,1, figsize=(12,13), sharex=True, sharey=True)
ax[0].plot(statemon.t[:]/ms, statemon.v[25*num_cols][:], 'C2', label='L3E')
ax[1].plot(statemon.t[:]/ms, statemon.v[57*num_cols][:], 'C3', label='L3I')
ax[2].plot(statemon.t[:]/ms, statemon.v[100*num_cols][:], 'C2', label='L5E')
ax[3].plot(statemon.t[:]/ms, statemon.v[142*num_cols][:], 'C3', label='L5I')
ax[4].plot(statemon.t[:]/ms, statemon.v[175*num_cols][:], 'C2', label='L6E')
ax[5].plot(statemon.t[:]/ms, statemon.v[215*num_cols][:], 'C3', label='L6I')
plt.ylim(-0.08, 0.02)
ax[2].set_ylabel('Membrame potential (v)')
ax[5].set_xlabel('Time (ms)')

plt.figure()
plt.plot(inputstatemon.t[:]/ms, inputstatemon.v[53*num_cols][:], 'C5', label='L6I')

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
#        sources = sources + 49
#    if src_group[k] == 'L5E':
#        sources = sources + 74
#    if src_group[k] == 'L5I':
#        sources = sources + 124
#    if src_group[k] == 'L6E':
#        sources = sources + 149
#    if src_group[k] == 'L6I':
#        sources = sources + 199
#    if src_group[k] == 'MTE':
#        sources = sources + 224
#    if src_group[k] == 'MTI':
#        sources = sources + 236
#    if src_group[k] == 'RI':
#        sources = sources + 242
#    if src_group[k] == 'SIE':
#        sources = sources + 249
#    if src_group[k] == 'PME':
#        sources = sources + 260
#        
#    if tgt_group[k] == 'L2/3E':
#        targets = targets
#    if tgt_group[k] == 'L2/3I':
#        targets = targets + 49
#    if tgt_group[k] == 'L5E':
#        targets = targets + 74
#    if tgt_group[k] == 'L5I':
#        targets = targets + 124
#    if tgt_group[k] == 'L6E':
#        targets = targets + 149
#    if tgt_group[k] == 'L6I':
#        targets = targets + 199
#    if tgt_group[k] == 'MTE':
#        targets = targets + 224
#    if tgt_group[k] == 'MTI':
#        targets = targets + 236
#    if tgt_group[k] == 'RI':
#        targets = targets + 242
#    if tgt_group[k] == 'SIE':
#        targets = targets + 249
#    if tgt_group[k] == 'PME':
#        targets = targets + 260
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

### Histograms of average firing rate ####
uniqueValues23E, occurCount23E = np.unique(spikemonL23E.i[index[0]:], return_counts=True)
uniqueValues23I, occurCount23I = np.unique(spikemonL23I.i[index[0]:], return_counts=True)
frequencies23E = occurCount23E/((duration/ms)/1000)
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.gca().set_title('L2/3E Average Firing')
plt.hist(frequencies23E,bins = 30)

uniqueValues5E, occurCount5E = np.unique(spikemonL5E.i[index[1]:], return_counts=True)
uniqueValues5I, occurCount5I = np.unique(spikemonL5I.i[index[0]:], return_counts=True)
frequencies5E = occurCount5E/((duration/ms)/1000)
# np.mean(frequencies5E)
plt.subplot(2,2,2)
plt.gca().set_title('L5E Average Firing')
plt.hist(frequencies5E,bins = 30)

uniqueValues6E, occurCount6E = np.unique(spikemonL6E.i[index[2]:], return_counts=True)
uniqueValues6I, occurCount6I = np.unique(spikemonL6I.i[index[0]:], return_counts=True)
frequencies6E = occurCount6E/((duration/ms)/1000)
plt.subplot(2,2,3)
plt.gca().set_title('L6E Average Firing')
plt.hist(frequencies6E,bins = 30)

#Cumulative Firing
L23E_firing = np.sum(occurCount23E)
print(L23E_firing)
L23I_firing = np.sum(occurCount23I)
print(L23I_firing)
L5E_firing = np.sum(occurCount5E)
print(L5E_firing)
L5I_firing = np.sum(occurCount5I)
print(L5I_firing)
L6E_firing = np.sum(occurCount6E)
print(L6E_firing)
L6I_firing = np.sum(occurCount6I)
print(L6I_firing)

total_firing = np.concatenate([spikemonL23E.t, spikemonL23I.t, spikemonL5E.t, spikemonL5I.t, spikemonL6E.t, spikemonL6I.t])

#Thalamus
uniqueValues_input, occurCount_input = np.unique(inputspikemon_TH.i[index[3]:], return_counts=True)
frequencies_input = (occurCount_input/((duration/ms)/1000))
plt.subplot(2,2,4)
plt.gca().set_title('Thalamus Average Firing')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Number of neurons')
plt.hist(frequencies_input,bins = 30, range = (0,20))

plt.figure(figsize=(6,4))
plt.hist(total_firing,bins = 1000)
plt.xlabel('time')
plt.ylabel('frequency')
plt.savefig('Plot.png', transparent=True)

#### Colourmaps ####
# top = cm.get_cmap('Oranges_r', 128)
# bottom = cm.get_cmap('Blues', 128)
# newcolours = np.vstack((top(np.linspace(0, 1, 128)),
#                       bottom(np.linspace(0, 1, 128))))
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

##### 3D spatial plot ####
#fig = plt.figure(figsize=(10,8))
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(columnsgroup_0.X/b2.umetre, columnsgroup_0.Y/b2.umetre, columnsgroup_0.Z/b2.umetre, marker='o')
#ax.scatter(columnsgroup_180.X/b2.umetre, columnsgroup_180.Y/b2.umetre, columnsgroup_0.Z/b2.umetre, marker='x')
#plt.xlabel("Distance (um)")
#plt.ylabel("Distance (um)")

end = time.time()
print('Time taken:', end-start)

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
