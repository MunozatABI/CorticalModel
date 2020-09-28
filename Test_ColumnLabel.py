# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:26:21 2020

@author: lmun373
"""
###############################################################################
########                   Import Libraries                             #######
###############################################################################
import brian2 as b2
from brian2 import mV, ms, ufarad, cm, umetre, volt, second, msiemens, siemens, nS, pA
import numpy as np
import time
import pandas as pd
import function_library as fl
import matplotlib.pyplot as plt
import re
import function_library as fl
import parameters
from parameters import *

from mpl_toolkits.mplot3d import Axes3D 

b2.start_scope()

start = time.time()

b2.prefs.codegen.target = 'numpy'

###############################################################################
########                   Parameter Definitions                        #######
###############################################################################

#Integration Parameters
simulation_time = 1000     # Total simulation time 

#Given current reversal potentials
EK = -90*b2.mV               # Potassium
ENa = 30*b2.mV               # Sodium
El = -10.6 * b2.mV
gl = 0.33

#Constant in threshold equation
C = 0.85

#Channel Parameters
tau1_AMPA = 0.5*ms
tau2_AMPA = 2.3*ms
Erev_AMPA = 0*mV
gpeak_AMPA = 0.1

tau1_GABAA = 0.5*ms
tau2_GABAA = 2*ms
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

#Read in synapse table
tab1 = pd.read_csv('Esser_table1.csv', nrows = 68, delimiter=' ', index_col=False) #Define input table

###############################################################################
########                     Neuron Equations                           #######
###############################################################################

###Equation Definitions
###From Esser, et al., 2005

eqs = '''
dtheta/dt = (-1*(theta - theta_eq)
             + C * (v - theta_eq)) / tau_theta
             : volt

I_syn = (v - Erev_AMPA) * g_AMPA + (v - Erev_NMDA) * g_NMDA + (v - Erev_GABAA) * g_GABAA + (v - Erev_GABAB) * g_GABAB : volt

dv/dt = ((-gNa*(v-ENa) - gK*(v-EK) - I_syn - gl*(v-El)))
        / tau_m    
        - int(v > theta) * int(t < (lastspike + t_spike)) * ((v - ENa) / (tau_spike))
              : volt
              
dx_AMPA/dt =  (-x_AMPA/tau2_AMPA) : 1

dg_AMPA/dt = ((tau2_AMPA / tau1_AMPA) ** (tau1_AMPA / (tau2_AMPA - tau1_AMPA))*x_AMPA-g_AMPA)/tau1_AMPA : 1

dx_NMDA/dt =  (-x_NMDA/tau2_NMDA) : 1

dg_NMDA/dt = ((tau2_NMDA / tau1_NMDA) ** (tau1_NMDA / (tau2_NMDA - tau1_NMDA))*x_NMDA-g_NMDA)/tau1_NMDA : 1

dx_GABAA/dt =  (-x_GABAA/tau2_GABAA) : 1

dg_GABAA/dt = ((tau2_GABAA / tau1_GABAA) ** (tau1_GABAA / (tau2_GABAA - tau1_GABAA))*x_GABAA-g_GABAA)/tau1_GABAA : 1

dx_GABAB/dt =  (-x_GABAB/tau2_GABAB) : 1

dg_GABAB/dt = ((tau2_GABAB / tau1_GABAB) ** (tau1_GABAB / (tau2_GABAB- tau1_GABAB))*x_GABAB-g_GABAB)/tau1_AMPA : 1

theta_eq : volt

tau_theta : second

tau_spike : second

t_spike : second

tau_m : second

gNa : 1

gK : 1

X : 1

Y : 1

Z : 1

neuron_type : integer (constant)

layer : integer (constant)

'''
MPE, MP5, MPI, THA = 1, 2, 3, 4

L23E, L23I, L5E, L5I, L6E, L6I, SIE, PME, MTE, MTI, RI  = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
            
neurons = b2.NeuronGroup(300, eqs,
          threshold = 'v > theta', 
          method = 'rk4',
          refractory = 'v > theta')

MTE_neurons = neurons[0:225]
input_neurons = neurons[225:]

neurons[:50].layer = L23E
neurons[50:75].layer = L23I
neurons[75:125].layer = L5E
neurons[125:150].layer = L5I
neurons[150:200].layer = L6E
neurons[200:225].layer = L6I
neurons[225:249].layer = SIE
neurons[249:261].layer = PME
neurons[261:275].layer = MTE
neurons[275:287].layer = MTI
neurons[287:300].layer = RI 

neurons.neuron_type['layer == L23E or layer == L6E'] = MPE
neurons.neuron_type['layer == L5E'] = MP5
neurons.neuron_type['layer == L23I or layer == L5I or layer == L6I'] = MPI
neurons.neuron_type['layer == SIE or layer == PME or layer == MTE or layer == MTI or layer == RI'] = THA


#### Initialiase Values

#Theta Eq
neurons.theta_eq['neuron_type == MPE or neuron_type == MP5'] = -53*mV
neurons.theta_eq['neuron_type == MPI or neuron_type == THA'] = -54*mV

#Tau Theta
neurons.tau_theta['neuron_type == MPE'] = 2.0*ms
neurons.tau_theta['neuron_type == MP5'] = 0.5*ms
neurons.tau_theta['neuron_type == MPI or neuron_type == THA'] = 1.0*ms

#Tau Spike
neurons.tau_spike['neuron_type == MPE'] = 1.75*ms
neurons.tau_spike['neuron_type == MP5'] = 0.6*ms
neurons.tau_spike['neuron_type == MPI or neuron_type == THA'] = 0.48*ms

#t Spike
neurons.t_spike['neuron_type == MPE'] = 2.0*ms
neurons.t_spike['neuron_type == MP5 or neuron_type == MPI or neuron_type == THA'] = 0.75*ms

#Tau m
neurons.tau_m['neuron_type == MPE'] = 15*ms
neurons.tau_m['neuron_type == MP5'] = 13*ms
neurons.tau_m['neuron_type == MPI or neuron_type == THA'] = 7*ms

#gNa Leak
neurons.gNa['neuron_type == MPE or neuron_type == MP5'] = 0.14
neurons.gNa['neuron_type == MPI or neuron_type == THA'] = 0.2

#gK Leak
neurons.gK['neuron_type == MPE or neuron_type == MPI or neuron_type == THA'] = 1.0
neurons.gK['neuron_type == MP5'] = 1.3

#Spiking Input Neurons
PMSI_spikes = np.ones(35)
THA_spikes = np.random.uniform(10, 20, 40)
input_rates = np.hstack([PMSI_spikes, THA_spikes])
Poisson_group = b2.PoissonGroup(75, [input_rates]*b2.hertz)

###############################################################################
########                          Synapses                              #######
###############################################################################
synapses_group = []

eqs_syn= '''
    w : 1
    '''

input_syn = b2.Synapses(Poisson_group, input_neurons, model=eqs_syn, on_pre='x_AMPA_post += w')
input_syn.connect(p=1)
input_syn.w = 1
synapses_group.append(input_syn)

syn_AMPA = b2.Synapses(neurons, neurons, model=eqs_syn, on_pre='x_AMPA_post += w')
syn_NMDA = b2.Synapses(neurons, neurons, model=eqs_syn, on_pre='x_NMDA_post += w')
syn_GABAA = b2.Synapses(neurons, neurons, model=eqs_syn, on_pre='x_GABAA_post += w')
syn_GABAB = b2.Synapses(neurons, neurons, model=eqs_syn, on_pre='x_GABAB_post += w')

for i, r in tab1.iterrows():
    src = re.sub('[(/)(0)(180)]', '', r.loc['SourceLayer'] + r.loc['SourceCellType'])
    tgt = re.sub('[(/)(0)(180)]', '', r.loc['TargetLayer'] + r.loc['TargetCellType'])
    if r.loc['Transmitter'] == 'AMPA':
        syn_AMPA.connect('layer_pre == {} and layer_post == {}'.format(src, tgt), p='{} * exp(-((X_pre-X_post)**2 + (Y_pre-Y_post)**2)/(2*(37.5*{})**2))'.format(r.loc['Pmax'], r.loc['Radius'])) #If it's this statement, it doesn't work
        syn_AMPA.w['layer_pre == {} and layer_post == {}'.format(src, tgt)] = (r.loc['Strength'])
        syn_AMPA.delay['layer_pre == {} and layer_post == {}'.format(src, tgt)] = r.loc['MeanDelay']*ms
        synapses_group.append(syn_AMPA)
    if r.loc['Transmitter'] == 'NMDA':
        syn_NMDA.connect('layer_pre == {} and layer_post == {}'.format(src, tgt), p=r.loc['Pmax']) #This statement works
        syn_NMDA.w['layer_pre == {} and layer_post == {}'.format(src, tgt)] = (r.loc['Strength'])
        syn_NMDA.delay['layer_pre == {} and layer_post == {}'.format(src, tgt)] = r.loc['MeanDelay']*ms
        synapses_group.append(syn_NMDA)
    if r.loc['Transmitter'] == 'GABAA':
        syn_GABAA.connect('layer_pre == {} and layer_post == {}'.format(src, tgt), p=r.loc['Pmax'])
        syn_GABAA.w['layer_pre == {} and layer_post == {}'.format(src, tgt)] = (r.loc['Strength'])
        syn_GABAA.delay['layer_pre == {} and layer_post == {}'.format(src, tgt)] = r.loc['MeanDelay']*ms
        synapses_group.append(syn_GABAA)
    if r.loc['Transmitter'] == 'GABAB':
        syn_GABAB.connect('layer_pre == {} and layer_post == {}'.format(src, tgt), p=r.loc['Pmax'])
        syn_GABAB.w['layer_pre == {} and layer_post == {}'.format(src, tgt)] = (r.loc['Strength'])
        syn_GABAB.delay['layer_pre == {} and layer_post == {}'.format(src, tgt)] = r.loc['MeanDelay']*ms
        synapses_group.append(syn_GABAB)

###############################################################################
########                         Monitors                               #######
###############################################################################
spikemon = b2.SpikeMonitor(neurons, variables='v', record=[1, 51, 76, 126, 151, 201, 226, 250, 262, 276, 288])
statemon = b2.StateMonitor(neurons, ['v', 'theta'], record=[1, 51, 76, 126, 151, 201, 226, 250, 262, 276, 288])
###############################################################################
########                         Run Model                              #######
###############################################################################
net = b2.Network(b2.collect())  #Automatically add visible objects 
net.add(synapses_group)           #Manually add list of synapses
net.run(simulation_time*b2.ms)

stop = time.time()

print('Time taken:', stop - start )

#####Plotting MPE neurons
fig, ax = plt.subplots(6, 1, figsize=(12,30), sharex = True)

ax[0].plot(statemon[1].t/b2.ms, statemon[1].v, 'C0-', label='L23E')
ax[1].plot(statemon[51].t/b2.ms, statemon[51].v, 'C1-', label='L23I')
ax[2].plot(statemon[76].t/b2.ms, statemon[76].v, 'C2-', label='L5E')
ax[3].plot(statemon[126].t/b2.ms, statemon[126].v, 'C3-', label='L5I')
ax[4].plot(statemon[151].t/b2.ms, statemon[151].v, 'C4-', label='L6E')
ax[5].plot(statemon[201].t/b2.ms, statemon[201].v, 'C5-', label='L6I')
ax[5].set_xlabel('Time (ms)')
ax[5].set_ylabel('v')
plt.legend();

#####Plotting Input neurons
fig, ax = plt.subplots(5, 1, figsize=(12,20), sharex = True)

ax[0].plot(statemon[226].t/b2.ms, statemon[226].v, 'C0-', label='SIE')
ax[1].plot(statemon[250].t/b2.ms, statemon[250].v, 'C1-', label='PME')
ax[2].plot(statemon[262].t/b2.ms, statemon[262].v, 'C2-', label='MTE')
ax[3].plot(statemon[276].t/b2.ms, statemon[276].v, 'C3-', label='MTI')
ax[4].plot(statemon[288].t/b2.ms, statemon[288].v, 'C4-', label='RI')
ax[4].set_xlabel('Time (ms)')
ax[4].set_ylabel('v')
plt.legend();