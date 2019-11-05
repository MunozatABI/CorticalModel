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
from mpl_toolkits.mplot3d import Axes3D 

import pandas as pd
import function_library as fl
import numpy as np
import time

b2.start_scope() #clear variables

###############################################################################
########                   Parameter Definitions                        #######
###############################################################################
tab1 = pd.read_csv('Esser_table1.csv', nrows = 69, delimiter=' ', index_col=False) #Define input table

#Simulation Parameters
duration = 700*ms     # Total simulation time
sim_dt = 0.1*ms           # Integrator/sampling step

tau1_AMPA = 0.5*ms
tau2_AMPA = 2.4*ms
Erev_AMPA = 0*mV
gpeak_AMPA = 0.1

tau1_GABAA = 1*ms
tau2_GABAA = 7*ms
Erev_GABAA = -70*mV
gpeak_GABAA = 0.33

tau1_GABAB = 60*ms
tau2_GABAB = 200*ms
Erev_GABAB = -90*mV
gpeak_GABAB = 0.0132

tau1_NMDA = 4*ms
tau2_NMDA = 40*ms
Erev_NMDA = 0*mV
gpeak_NMDA = 0.1

#Constants
EK = -90*mV               # Potassium
ENa = 30*mV               # Sodium
El = -10.6 * mV
gl = 0.33

#Constant in threshold equation
C = 0.85

###############################################################################
########                     Neuron Equations                           #######
###############################################################################
###From Esser, et al., 2005

eqs = fl.equation('current')
eqs += 'I_syn = (v - Erev_AMPA)*g_AMPAa +' + '+'.join(['(v - Erev_{}) * g_{}{}'.format(r.loc['Transmitter'],r.loc['Transmitter'],i) for i,r in tab1.iterrows()]) + ' : volt\n'
eqs += 'g_AMPAa : 1\n ' + ''.join(['g_{}{} : 1\n'.format(r.loc['Transmitter'], i) for i,r in tab1.iterrows()])

###############################################################################
########                      Create Neurons                            #######
###############################################################################
#Motor Cortex Column
MP_neurons = 225 # Number of neurons                                                                     
ntype = ['L3E', 'L3I', 'L5E', 'L5I', 'L6E', 'L6I']                 # Types of neurons
num = [50, 25, 50, 25, 50, 25]                                # Number of each type of neuron
column1 = fl.generate.neurons(num, ntype, eqs)

#Input Areas - SMA, PME, THA, RN
in_type = ['THA']
in_num = [37]
Input_Neurons = fl.generate.neurons(in_num, in_type, eqs)

times = []
indices = []
#num_spikes1 = round(duration/b2.ms/1000*15)

for j in range(25):
    x = 0
    numspikes = np.random.uniform(10, 20, 1)
    s1 = np.random.uniform(50, 100, int(round(numspikes[0])))
    for k in range(len(s1)):
        x += s1[k]
        times.append(round(x))
        indices.append(j)

mu, sigma = round(1000/1), round(100/1.) 
num_spikes2 = round(duration/b2.ms/1000*1)

for j in range(12):
    x = 0
    s2 = np.random.normal(mu, sigma, num_spikes2)
    for k in range(len(s2)):
        x += s2[k]
        times.append(round(x))
        indices.append(j + 24)

input_indices= b2.array(indices)
input_times = times*ms
Spike = b2.SpikeGeneratorGroup(37, input_indices, input_times)

### Define Neuronal Subgroups
neuron_group = {'L2/3E': column1[0:50], 
                'L2/3I': column1[50:75], 
                'L5E': column1[75:125], 
                'L5I': column1[125:150], 
                'L6E': column1[150:200], 
                'L6I': column1[200:225],
                'MTE': Input_Neurons[0:12],
                'MTI': Input_Neurons[12:18],
                'RI': Input_Neurons[18:25],
                'SIE': Input_Neurons[25:31],
                'PME': Input_Neurons[31:37]
                }

#TMS = b2.SpikeGeneratorGroup(1, [0], [500]*ms)
###############################################################################
########                          Synapses                              #######
###############################################################################
Input_synapses = fl.generate.synapses([Spike], [Input_Neurons], ['AMPA'], [1], [1], [1.4])
all_synapses = fl.generate.model_synapses(tab1, neuron_group)

#Model of TMS activation
#TMS_synapse = b2.Synapses(TMS, column1, eqs_syn.format(tr='AMPA',st = 'b'), method = 'rk4', on_pre='x_{}{} += w'.format('AMPA', 'b'), delay = 1.4*ms)
#TMS_synapse.connect(p=1)
#TMS_synapse.w = 1

###############################################################################
########                         Monitors                               #######
###############################################################################

statemon = b2.StateMonitor(column1, 'v', record=range(225))
spikemon = b2.SpikeMonitor(column1, variables = ['v', 't'])
spikemonL23 = b2.SpikeMonitor(neuron_group['L2/3E'], variables = ['v', 't'])
spikemonL5 = b2.SpikeMonitor(neuron_group['L5E'], variables = ['v', 't'])
spikemonL6 = b2.SpikeMonitor(neuron_group['L6E'], variables = ['v', 't'])
inputstatemon = b2.StateMonitor(Input_Neurons, 'v', record=range(37))
inputspikemon = b2.SpikeMonitor(neuron_group['MTE'], variables = ['v', 't'])

###############################################################################
########                         Run Model                              #######
###############################################################################
net = b2.Network(b2.collect())  #Automatically add visible objects 
net.add(Input_synapses, all_synapses)           #Manually add list of synapses

#Check initial v
print('Before v = %s' % column1.v[0])
start = time.time() #Running time

net.run(duration) #Run

end = time.time()

#See what the final v is
print('After v= %s' % column1.v[0])

print('Time taken:', end-start)

###############################################################################
########                       Plot Graphs                              #######
###############################################################################
#Look at data after 200ms
timearrays = [spikemonL23.t, spikemonL5.t, spikemonL6.t, inputspikemon.t, spikemon.t]
index = [0, 0, 0, 0, 0]
for j in range(len(timearrays)):
    for i in range(len(timearrays[j])):
        if timearrays[j][i] < 200*ms:
            index[j] = i 

#Plot Cortex Membrane Potentials
plt.figure(figsize=(12, 5))
plt.subplot(2,1,1)
plt.plot(statemon.t[2000:]/ms, statemon.v[25][2000:], 'C0', label='L3E')
plt.plot(statemon.t[2000:]/ms, statemon.v[57][2000:], 'C1', label='L3I')
plt.plot(statemon.t[2000:]/ms, statemon.v[100][2000:], 'C2', label='L5E')
plt.plot(statemon.t[2000:]/ms, statemon.v[142][2000:], 'C3', label='L5I')
plt.plot(statemon.t[2000:]/ms, statemon.v[175][2000:], 'C4', label='L6E')
plt.plot(statemon.t[2000:]/ms, statemon.v[215][2000:], 'C5', label='L6I')
plt.ylabel('v')
plt.legend()
plt.subplot(2,1,2)
plt.plot(spikemon.t[index[4]:]/b2.ms, spikemon.i[index[4]:], '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron')
plt.show()

#Plot Thalamus Membrane Potential
plt.figure(figsize=(12,5))
plt.subplot(2,1,1)
plt.plot(inputstatemon.t[2000:]/ms, inputstatemon.v[1][2000:], 'C6', label='MTE')
#plt.plot(inputstatemon.t[2000:]/ms, inputstatemon.v[120][2000:], 'C4', label='PME')
plt.ylabel('v')
plt.legend()
plt.subplot(2,1,2)
plt.plot(inputspikemon.t[index[3]:]/b2.ms, inputspikemon.i[index[3]:], '.k')

###Histograms of average membrane potential
a = []
b = []
for i in range(225):
    a.append(np.mean([statemon.v[i][2000:]]))
    
for i in range(25):
    b.append(np.mean([inputstatemon.v[i][2000:]]))

plt.figure(figsize=(12,8))
plt.subplot(2,2,1).set_title('L2/3E Membrane potential')
plt.hist(a[0:50],bins = 30) #L2/3E
plt.subplot(2,2,2).set_title('L5E Membrane potential')
plt.hist(a[75:125],bins = 30) #L5E
plt.subplot(2,2,3).set_title('L6E Membrane potential')
plt.hist(a[150:200],bins = 30) #L6E
plt.subplot(2,2,4).set_title('Excitatory Thalamus Membrane potential')
plt.hist(b,bins = 30) #L6E

###Histograms of average firing rate
uniqueValues23, occurCount23 = np.unique(spikemonL23.i[index[0]:], return_counts=True)
frequencies23 = occurCount23/((duration/ms)/1000)
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.gca().set_title('L2/3E Average Firing')
plt.hist(frequencies23,bins = 30)

uniqueValues5, occurCount5 = np.unique(spikemonL5.i[index[1]:], return_counts=True)
frequencies5 = occurCount5/((duration/ms)/1000)
plt.subplot(2,2,2)
plt.gca().set_title('L5E Average Firing')
plt.hist(frequencies5,bins = 30)

uniqueValues6, occurCount6 = np.unique(spikemonL6.i[index[2]:], return_counts=True)
frequencies6 = occurCount6/((duration/ms)/1000)
plt.subplot(2,2,3)
plt.gca().set_title('L6E Average Firing')
plt.hist(frequencies6,bins = 30)

#Thalamus
uniqueValues_input, occurCount_input = np.unique(inputspikemon.i[index[3]:], return_counts=True)
frequencies_input = (occurCount_input/((duration/ms)/1000))/2
plt.subplot(2,2,4)
plt.gca().set_title('Thalamus Average Firing')
plt.hist(frequencies_input,bins = 30)

###Colourmaps
top = cm.get_cmap('Oranges_r', 128)
bottom = cm.get_cmap('Blues', 128)
newcolours = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
colours = mpl.colors.ListedColormap(newcolours, name='OrangeBlue')

viridis = cm.get_cmap('viridis', 256)

#Colourmap of Cortex
data = statemon.v
fig, axs = plt.subplots(figsize=(5, 2), constrained_layout=True)
psm = axs.pcolormesh(data, cmap=viridis, rasterized=True, vmin=-0.07, vmax=-0.05)
fig.colorbar(psm, ax=axs)
plt.show()

#Colourmap of Input Areas
data = inputstatemon.v
fig, axs = plt.subplots(figsize=(5, 2), constrained_layout=True)
psm = axs.pcolormesh(data, cmap=viridis, rasterized=True, vmin=-0.07, vmax=-0.05)
fig.colorbar(psm, ax=axs)
plt.show()

#3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(column1.X/b2.umetre, column1.Y/b2.umetre, column1.Z/b2.umetre, marker='o')
###############################################################################
########                       Unused Code                              #######
###############################################################################
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