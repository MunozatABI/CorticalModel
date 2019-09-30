# -*- coding: utf-8 -*-
"""Column_Esser.ipynb
In this script, I am aiming to make a column of 225 Neurons
"""
###############################################################################
########                   Import Libraries                             #######
###############################################################################

import brian2 as b2
from brian2 import mV, ms
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

import pandas as pd
import function_library as fl
import numpy as np

b2.start_scope() #clear variables

###############################################################################
########                   Parameter Definitions                        #######
###############################################################################
tab1 = pd.read_csv('Esser_table1.csv', nrows = 69, delimiter=' ', index_col=False) #Define input table

#Sinulation Parameters
duration = 700*ms     # Total simulation time
sim_dt = 0.1*ms           # Integrator/sampling step

fl.parameters

###############################################################################
########                     Neuron Equations                           #######
###############################################################################
###From Esser, et al., 2005

eqs = ''
eqs = fl.equation(eqs, 'current')

eqs += 'I_syn = (v - Erev_AMPA)*g_AMPAa + (v - Erev_AMPA)*g_AMPAb +' + '+'.join(['(v - Erev_{}) * g_{}{}'.format(r.loc['Transmitter'],r.loc['Transmitter'],i) for i,r in tab1.iterrows()]) + ' : volt\n'
eqs += ''.join(['g_{}{} : 1\n'.format(r.loc['Transmitter'], i) for i,r in tab1.iterrows()])

###############################################################################
########                      Create Neurons                            #######
###############################################################################

#Motor Cortex
MP_neurons = 225 # Number of neurons                                                                     
ntype = ['MPE', 'MPI', 'L5E', 'MPI', 'MPE', 'MPI']                 # Types of neurons
num = [50, 25, 50, 25, 50, 25]                                # Number of each type of neuron
initial_values={} 

column1 = b2.NeuronGroup(MP_neurons, model = eqs, 
                  threshold = 'v >= theta', 
                  reset = 'v = theta_eq',
                  events={'on_spike': 'v >= theta'},
                  method = 'rk4',
                  refractory = 2*ms)

fl.initialise_neurons(ntype, num, initial_values)
column1.set_states(initial_values) #Apply initial variables to group
column1.run_on_event('on_spike', 'count = count + 1') #On the event of a spike, the g_spike part of the equation turns on and after t_spike, it turns off. This counter helps to turn the function off after t_spike has passed.
column1.v = column1.theta_eq #initialise resting potential

#Input Areas - SMA, PME, THA, RN
in_type = ['THA']
in_num = [125]
in_values = {}
fl.initialise_neurons(in_type, in_num, in_values)

Input_Neurons = b2.NeuronGroup(125, eqs,
                  threshold = 'v > theta', 
                  reset = 'v = theta',
                  events={'on_spike': 'v > theta'},
                  method = 'rk4',
                  refractory = 2*ms)

Input_Neurons.set_states(in_values)
Input_Neurons.v = Input_Neurons.theta_eq #initialise resting potential
Input_Neurons.run_on_event('on_spike', 'count = count + 1')

times = []
indices = []
num_spikes1 = round(duration/b2.ms/1000*15)

for j in range(75):
    x = 0
    s1 = np.random.uniform(50, 100, num_spikes1)
    for k in range(len(s1)):
        x += s1[k]
        times.append(round(x))
        indices.append(j)

mu, sigma = round(1000/1), round(100/1) 
num_spikes2 = round(duration/b2.ms/1000*1)

for j in range(50):
    x = 0
    s2 = np.random.normal(mu, sigma, num_spikes2)
    for k in range(len(s2)):
        x += s2[k]
        times.append(np.round(x))
        indices.append(j + 74)

input_indices= b2.array(indices)
input_times = times*ms
Spike = b2.SpikeGeneratorGroup(125, input_indices, input_times)

### Define Neuronal Subgroups
neuron_group = {'L2/3E': column1[0:50], 
                'L2/3I': column1[50:75], 
                'L5E': column1[75:125], 
                'L5I': column1[125:150], 
                'L6E': column1[150:200], 
                'L6I': column1[200:225],
                'MTE': Input_Neurons[0:25],
                'MTI': Input_Neurons[25:50],
                'RI': Input_Neurons[50:75],
                'SIE': Input_Neurons[75:100],
                'PME': Input_Neurons[100:125]
                }

TMS = b2.SpikeGeneratorGroup(1, [0], [500]*ms)
###############################################################################
########                          Synapses                              #######
###############################################################################
#General Model for synapse equations 
#eqs_syn from https://brian2.readthedocs.io/en/stable/user/converting_from_integrated_form.html
#Biexponential synapse
all_synapses = [] #Create list structure to store synapse information
eqs_syn= '''
    dg_{tr}_syn{st}/dt = ((tau2_{tr} / tau1_{tr}) ** (tau1_{tr} / (tau2_{tr} - tau1_{tr}))*x_{tr}{st}-g_{tr}_syn{st})/tau1_{tr} : 1
    dx_{tr}{st}/dt =  (-x_{tr}{st}/tau2_{tr}) : 1
    g_{tr}{st}_post = g_{tr}_syn{st} : 1 (summed)
    w : 1
    '''
#Input synapses
synapse = b2.Synapses(Spike, Input_Neurons, eqs_syn.format(tr = 'AMPA',st = 'a'), method = 'rk4', on_pre='x_{}{} += w'.format('AMPA', 'a'), delay = 1.4*ms)
synapse.connect(j='i', p=1)
synapse.w = 1
#plt.figure(figsize = (40,40))
#fl.visualise_connectivity(synapse)

#Define motor cortex synapses
for i, r in tab1.iterrows():
        
    src = r.loc['SourceLayer'] + re.sub('[018()]', '', r.loc['SourceCellType'])
    tgt = r.loc['TargetLayer'] + re.sub('[018()]', '', r.loc['TargetCellType'])
    syn = b2.Synapses(neuron_group[src],
                      neuron_group[tgt],
                      model = eqs_syn.format(tr = r.loc['Transmitter'],st = i),
                      method = 'rk4',
                      on_pre='x_{}{} += w'.format(r.loc['Transmitter'], i))
    
    syn.connect(p=r.loc['Pmax']) #Probability of connecting 
    syn.w = (r.loc['Strength']/10)  #Weights
    syn.delay = r.loc['MeanDelay']*ms
    all_synapses.append(syn)

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
inputstatemon = b2.StateMonitor(Input_Neurons, 'v', record=range(125))
inputspikemon = b2.SpikeMonitor(neuron_group['MTE'], variables = ['v', 't'])

###############################################################################
########                         Run Model                              #######
###############################################################################
net = b2.Network(b2.collect())  #Automatically add visible objects 
net.add(all_synapses)           #Manually add list of synapses

#Check initial v
print('Before v = %s' % column1.v[0])

net.run(duration) #Run

#See what the final v is
print('After v= %s' % column1.v[0])

###############################################################################
########                       Plot Graphs                              #######p
###############################################################################
#Look at data after 200ms
timearrays = [spikemonL23.t, spikemonL5.t, spikemonL6.t, inputspikemon.t, spikemon.t]
index = [0, 0, 0, 0, 0]
for j in range(len(timearrays)):
    for i in range(len(timearrays[j])):
        if timearrays[j][i] < 200*ms:
            index[j] = i 

#Plot Coretx
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

#Plot Thalamus Input
plt.figure(figsize=(12,5))
plt.subplot(2,1,1)
plt.plot(inputstatemon.t[2000:]/ms, inputstatemon.v[1][2000:], 'C6', label='MTE')
plt.plot(inputstatemon.t[2000:]/ms, inputstatemon.v[120][2000:], 'C4', label='PME')
plt.ylabel('v')
plt.legend()
plt.subplot(2,1,2)
plt.plot(inputspikemon.t[index[3]:]/b2.ms, inputspikemon.i[index[3]:], '.k')

#Histograms of average membrane potential
a = []
b = []
for i in range(225):
    a.append(np.mean([statemon.v[i][2000:]]))
    
for i in range(25):
    b.append(np.mean([inputstatemon.v[i][2000:]]))

plt.figure(figsize=(12,8))
plt.subplot(2,2,1).set_title('L2/3I Membrane potential')
plt.hist(a[0:50],bins = 30) #L2/3E
plt.subplot(2,2,2).set_title('L5I Membrane potential')
plt.hist(a[75:125],bins = 30) #L5E
plt.subplot(2,2,3).set_title('L6I Membrane potential')
plt.hist(a[150:200],bins = 30) #L6E
plt.subplot(2,2,4).set_title('Excitatory Thalamus Membrane potential')
plt.hist(b,bins = 30) #L6E

#Histograms of average firing rate
uniqueValues23, occurCount23 = np.unique(spikemonL23.i[index[0]:], return_counts=True)
frequencies23 = occurCount23/((duration/ms)/1000)
plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.gca().set_title('L2/3I Average Firing')
plt.hist(frequencies23,bins = 30)

uniqueValues5, occurCount5 = np.unique(spikemonL5.i[index[1]:], return_counts=True)
frequencies5 = occurCount5/((duration/ms)/1000)
plt.subplot(2,2,2)
plt.gca().set_title('L5I Average Firing')
plt.hist(frequencies5,bins = 30)

uniqueValues6, occurCount6 = np.unique(spikemonL6.i[index[2]:], return_counts=True)
frequencies6 = occurCount6/((duration/ms)/1000)
plt.subplot(2,2,3)
plt.gca().set_title('L6I Average Firing')
plt.hist(frequencies6,bins = 30)

##Thalamus
uniqueValues_input, occurCount_input = np.unique(inputspikemon.i[index[3]:], return_counts=True)
frequencies_input = occurCount_input/((duration/ms)/1000)
plt.subplot(2,2,4)
plt.gca().set_title('Thalamus Average Firing')
plt.hist(frequencies_input,bins = 30)

#Colourmap of Cortex
top = cm.get_cmap('Oranges_r', 128)
bottom = cm.get_cmap('Blues', 128)
newcolours = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
colours = mpl.colors.ListedColormap(newcolours, name='OrangeBlue')

viridis = cm.get_cmap('viridis', 256)

data = statemon.v
fig, axs = plt.subplots(figsize=(5, 2), constrained_layout=True)
psm = axs.pcolormesh(data, cmap=viridis, rasterized=True, vmin=-0.07, vmax=-0.05)
fig.colorbar(psm, ax=axs)
plt.show()
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