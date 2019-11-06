# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:12:50 2019

@author: lmun373
"""
###############################################################################
########                   Import Libraries                             #######
###############################################################################
import brian2 as b2
from brian2 import mV, ms, ufarad, cm, umetre, volt, second, msiemens, siemens, nS, pA
import numpy as np
import function_library as fl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
b2.start_scope()

###############################################################################
########                   Parameter Definitions                        #######
###############################################################################
import parameters as pm
from parameters import * 

simulation_time = 1000     # Total simulation time 

###############################################################################
########                     Neuron Equations                           #######
###############################################################################
Transmitters = ['AMPA', 'AMPA', 'GABAB']
eqs = fl.equation('current')
eqs += 'I_syn = (v - Erev_AMPA) * g_AMPA + (v - Erev_AMPA) * g_AMPA_2 +' + '+'.join(['(v - Erev_{}) * g_{}{}'.format(Transmitters[i],Transmitters[i],i) for i in range(len(Transmitters))]) + ' : volt\n'
eqs += 'g_AMPA : 1\n g_AMPA_2 :1\n' + ''.join(['g_{}{} : 1\n'.format(Transmitters[i], i) for i in range(len(Transmitters))])

###############################################################################
########                      Create Neurons                            #######
###############################################################################
#initialise neurons
neuron_type = ['MPE', 'L5E', 'MPI']
neuron_num = [1, 1, 1]
Neurons = fl.generate.neurons(neuron_num, neuron_type, eqs)

#initialise input (thalamic) neurons
in_type = ['THA']
in_num = [2]
Input_Neurons = fl.generate.neurons(in_num, in_type, eqs)

#Spike definitions for input neurons
Spike1 = b2.SpikeGeneratorGroup(1, [0], [500]*ms) 
Spike2 = b2.SpikeGeneratorGroup(1, [0], [500]*ms)

#Subgrouping definitions
A = Input_Neurons[0:1] #Input Neuron 1
B = Input_Neurons[1:2] #Input Neuron 2
G = Neurons[0:1] #MP Excitatory
H = Neurons[1:2] #L5 Excitatory
K = Neurons[2:3] #MP Inhibitory

###############################################################################
########                          Synapses                              #######
###############################################################################
#Input_synpase definitions
Input_syn1 = b2.Synapses(Spike1, A, '''
dg_AMPA_syn/dt = ((tau2_AMPA / tau1_AMPA) ** (tau1_AMPA / (tau2_AMPA - tau1_AMPA))*x_AMPA-g_AMPA_syn)/tau1_AMPA : 1
dx_AMPA/dt =  (-x_AMPA/tau2_AMPA) : 1
g_AMPA_post = g_AMPA_syn : 1 (summed)
w : 1
''', on_pre='x_AMPA += w')    
Input_syn1.connect(p = 1)
Input_syn1.w = 1

Input_syn2 = b2.Synapses(Spike2, B, '''
dg_AMPA_syn_2/dt = ((tau2_AMPA / tau1_AMPA) ** (tau1_AMPA / (tau2_AMPA - tau1_AMPA))*x_AMPA_2-g_AMPA_syn_2)/tau1_AMPA : 1
dx_AMPA_2/dt =  (-x_AMPA_2/tau2_AMPA) : 1
g_AMPA_2_post = g_AMPA_syn_2 : 1 (summed)
w : 1
''', on_pre='x_AMPA_2 += w')    
Input_syn2.connect(p = 1)
Input_syn2.w = 1

synapses_group = []
synapses_group.append(Input_syn1)
synapses_group.append(Input_syn2)

Inputs = [A] ### Define Inputs here ###
Targets = [G]  ### Define Targets here ###
prob = [1]
weight = [0.1]
delay = [1.4]

synapses_group = fl.generate.synapses(Inputs, Targets, Transmitters, prob, weight, delay, S=True)
###############################################################################
########                         Monitors                               #######
###############################################################################
#Monitoring membrane potentials
M1 = b2.StateMonitor(Inputs[0], ['v', 'theta'], record=True)
M2 = b2.StateMonitor(Targets[0], 'v', record=True)
SpikeMon1 = b2.SpikeMonitor(Spike1)
SpikeMon = b2.SpikeMonitor(A)
#SpikeMon2 = b2.SpikeMonitor(Spike2)

###############################################################################
########                         Run Model                              #######
###############################################################################
net = b2.Network(b2.collect())  #Automatically add visible objects 
net.add(synapses_group)           #Manually add list of synapses

net.run(simulation_time*ms) #Run simulation

###############################################################################
########                       Plot Graphs                              #######
###############################################################################
#Plot Membrane Potential
Monitors = [M1, M2]
Labels = ['Input Neuron 1', 'Target Neuron 1']
fl.visualise.membrane_voltage(Labels, Monitors, [0, 0])
#b2.plot(M1.t/b2.ms, M1.theta[0], 'C4-')
#b2.plot(SpikeMon.t/b2.ms, SpikeMon.i, '.k') #Plot spiking

#fl.visualise.average_firing(['Average Firing'], [SpikeMon], simulation_time)

###############################################################################
########                       Unused Code                              #######
###############################################################################
#3D plot
#b2.figure()
#b2.plot(Neurons.x / b2.umetre, Neurons.y / b2.umetre, 'og')
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(Neurons.x/b2.umetre, Neurons.z/b2.umetre, Neurons.z/b2.umetre, marker='o')

#Initialising neurons with spatial dimensions
#initial_values = {'theta_eq': [-54, -53, -53]*mV,            #resting threshold
#                   'tau_theta': [1, 2, 2]*ms,              #threshold time constant
#                   'tau_spike': [0.48, 1.75, 1.75]*ms,        #time constant during spike
#                   't_spike': [0.75, 2, 2]*ms,          #length of time of spike
#                   'tau_m': [7, 15, 15]*ms,                  #membrane time constant
#                   'gNa': [0.2, 0.14, 0.14],                 #sodium leak
#                   'gK': [1.0, 1.0, 1.0],
#                   'count':[0,0,0],
#                   'x': [5, 10, 15]*umetre,
#                   'y': [3, 6, 9]*umetre,
#                   'z': [2, 4, 6]*umetre}                 #potassium leak

#eqs += '''
#x : meter
#y : meter
#z : meter
#'''
#b2.figure()
#b2.plot(M1.t[10000:10050]/b2.ms, M1.v[0][10000:10050], 'C1', label='v')
#b2.plot(M1.t[10000:10050]/b2.ms, M1.theta[0][10000:10050], 'C4-', label = 'theta')
#b2.plot(SpikeMon.t/b2.ms, SpikeMon.i, '.k') #Plot spiking