# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:12:50 2019

@author: lmun373
"""

import brian2 as b2
from brian2 import mV, ms, ufarad, cm, umetre, volt, second
import numpy as np
import function_library as fl
import matplotlib.pyplot as plt

b2.start_scope()

#tau = 10*b2.ms
#simple_eqs = '''
#dv/dt = (1-v)/tau : 1
#'''

#Integration Parameters
simulation_time = 1500     # Total simulation time 

#Given current reversal potentials
EK = -90*b2.mV               # Potassium
ENa = 30*b2.mV               # Sodium
El = -10.6 * b2.mV
gl = 0.33
#Constant in threshold equation
C = 0.85

#VT = -53*mV
#Channel Parameters
tau1_AMPA = 0.5*ms
tau2_AMPA = 2.3*ms
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

Transmitters = ['AMPA', 'AMPA', 'GABAB']

###Equation Definitions
###From Esser, et al., 2005

eqs = '''
dtheta/dt = (-1*(theta - theta_eq)
             + C * (v - theta_eq)) / tau_theta
             : volt

dv/dt = ((-gNa*(v-ENa) - gK*(v-EK) - I_syn - gl*(v-El)))
        / tau_m    
        - int(v >= theta) * int(t < (start_time + t_spike)) * (v - ENa) / tau_spike
          : volt

start_time = t * int(v>=theta) - count*dt : second

theta_eq : volt

tau_theta : second

tau_spike : second

t_spike : second

tau_m : second

gNa : 1

gK : 1

count : 1

g_AMPA : 1

g_AMPA_2 : 1

'''

#I_int = I_h + I_t + I_nap : volt
#
#I_h = (gpeak_GABAB)  * ((1/(1 + exp((v/volt - (-75.0))/5.5))) ** n) * h * (v - ENa) : volt
#
#I_t = (gpeak_GABAB) * (1/(1 + exp((-v/volt + 59.0)/6.2)) ** n) * (1/(1 + exp((v/volt + 83.0)/4.0))) * (v - ENa) : volt
#
#I_nap = (gpeak_GABAB) * (1/(1 + exp(-v/volt + 55.7)/7.7)  ** n) * (v - ENa) : volt
#
#dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/
#    ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
#
#dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
#    (exp((15.*mV-v+VT)/(5.*mV))-1.)/
#    ms*(1.-n)-.5*exp((10.*mV-v+VT)/
#    (40.*mV))/ms*n : 1 

eqs += 'I_syn = (v - Erev_AMPA) * g_AMPA + (v - Erev_AMPA) * g_AMPA_2 +' + '+'.join(['(v - Erev_{}) * g_{}{}'.format(Transmitters[i],Transmitters[i],i) for i in range(len(Transmitters))]) + ' : volt\n'
eqs += ''.join(['g_{}{} : 1\n'.format(Transmitters[i], i) for i in range(len(Transmitters))])

#initialise variables
initial_values = {'theta_eq': [-54, -53, -53]*mV,            #resting threshold
                   'tau_theta': [1, 2, 2]*ms,              #threshold time constant
                   'tau_spike': [0.48, 1.75, 1.75]*ms,        #time constant during spike
                   't_spike': [0.75, 2, 2]*ms,          #length of time of spike
                   'tau_m': [7, 15, 15]*ms,                  #membrane time constant
                   'gNa': [0.2, 0.14, 0.14],                 #sodium leak
                   'gK': [1.0, 1.0, 1.0]}                 #potassium leak

synapses_group = []
mean_frequency = 1
#Spiking Input Neurons

#Spike = b2.NeuronGroup(1, '''dv/dt = 0.25*(xi*sqrt(second)+4)*Hz : 1''', 
#                                      threshold='v>1', reset='v=0', method='euler')

in_type = ['THA']
in_num = [2]
in_values = {}
fl.initialise_neurons(in_type, in_num, in_values)

Input_Neurons = b2.NeuronGroup(2, eqs,
                  threshold = 'v >= theta', 
                 reset = 'v = theta',
                  events={'on_spike': 'v >= theta'},
                  method = 'euler')

Input_Neurons.set_states(in_values)
Input_Neurons.v = Input_Neurons.theta_eq #initialise resting potential
Input_Neurons.run_on_event('on_spike', 'count = count + 1')

A = Input_Neurons[0:1]
B = Input_Neurons[1:2]

mu, sigma = round(1000/mean_frequency), round(100/mean_frequency) 
num_spikes = round(simulation_time/mean_frequency)
s = np.random.normal(mu, sigma, num_spikes)
x = 0
times = []
zeros = []

for i in range(len(s)):
    x += s[i]
    times.append(round(x))
    zeros.append(0)

indices = b2.array(zeros)
times = b2.array(times)*ms
Spike = b2.SpikeGeneratorGroup(1, indices, times)
indices_1 = [0]
times_1 = [1000]*ms
indices_2 = [0]
times_2 = [1000]*ms
Spike1 = b2.SpikeGeneratorGroup(1, indices_1, times_1)
Spike2 = b2.SpikeGeneratorGroup(1, indices_2, times_2)

Input_syn1 = b2.Synapses(Spike1, A, '''
dg_AMPA_syn/dt = ((tau2_AMPA / tau1_AMPA) ** (tau1_AMPA / (tau2_AMPA - tau1_AMPA))*x_AMPA-g_AMPA_syn)/tau1_AMPA : 1
dx_AMPA/dt =  (-x_AMPA/tau2_AMPA) : 1
g_AMPA_post = g_AMPA_syn : 1 (summed)
w : 1
''', on_pre='x_AMPA += w')    
Input_syn1.connect(p = 1)
Input_syn1.w = 10

Input_syn2 = b2.Synapses(Spike2, B, '''
dg_AMPA_syn_2/dt = ((tau2_AMPA / tau1_AMPA) ** (tau1_AMPA / (tau2_AMPA - tau1_AMPA))*x_AMPA_2-g_AMPA_syn_2)/tau1_AMPA : 1
dx_AMPA_2/dt =  (-x_AMPA_2/tau2_AMPA) : 1
g_AMPA_2_post = g_AMPA_syn_2 : 1 (summed)
w : 1
''', on_pre='x_AMPA_2 += w')    
Input_syn2.connect(p = 1)
Input_syn2.w = 10

synapses_group.append(Input_syn1)

Neurons = b2.NeuronGroup(3, eqs,
                  threshold = 'v >= theta', 
                 reset = 'v = theta',
                  events={'on_spike': 'v >= theta'},
                  method = 'euler')

#fl.initialise_neurons(ntype, transmitter, num, initial_values)

Neurons.set_states(initial_values)

Neurons.v = Neurons.theta_eq #initialise resting potential

Neurons.run_on_event('on_spike', 'count = count + 1')

#Neurons.I_int = 0*mV

#Subgrouping

G = Neurons[0:1] #MP Inhibitory
H = Neurons[1:2] #MP Excitatory
K = Neurons[2:3] #L5 Excitatory

Inputs = [A, B]
Targets = [K, K]

eqs_syn= '''
    dg_{tr}_syn{st}/dt = ((tau2_{tr} / tau1_{tr}) ** (tau1_{tr} / (tau2_{tr} - tau1_{tr}))*x_{tr}{st}-g_{tr}_syn{st})/tau1_{tr} : 1
    dx_{tr}{st}/dt =  (-x_{tr}{st}/tau2_{tr}) : 1
    g_{tr}{st}_post = g_{tr}_syn{st} : 1 (summed)
    w : 1
    '''
    
##Putting Synapses into a list structure

for i in range(len(Inputs)):
    syn = b2.Synapses(Inputs[i], Targets[i], eqs_syn.format(tr = Transmitters[i],st = i), on_pre='x_{}{} += w'.format(Transmitters[i], i))
    syn.connect(p=1)
    syn.w = 1
    synapses_group.append(syn)

M1 = b2.StateMonitor(K, ['v', 'theta'], record=True)
M2 = b2.StateMonitor(K, 'v', record=True)
MS = b2.StateMonitor(Input_Neurons, 'v', record=range(2))
SpikeMon = b2.SpikeMonitor(Input_Neurons)
#SpikeMon2 = b2.SpikeMonitor(Spike2)

net = b2.Network(b2.collect())  #Automatically add visible objects 
net.add(synapses_group)           #Manually add list of synapses
net.run(simulation_time*ms)

#Plotting 3 neurons
b2.figure(figsize=(12,4))
b2.plot(M1.t/b2.ms, M1.v[0], 'C0-', label='neuron1')
b2.plot(M1.t/b2.ms, M1.theta[0], 'C1-', label='theta')
#b2.plot(M2.t/b2.ms, M2.v[0], 'C2-', label='neuron2')
b2.xlabel('Time (ms)')
b2.ylabel('v')
b2.legend();

#Plot Firing
b2.figure(figsize=(12,4))
b2.plot(MS.t/b2.ms, MS.v[0], 'C1-', label='Input')
#b2.plot(MS.t/b2.ms, MS.v[1], 'C2-', label='Input')
b2.plot(SpikeMon.t/b2.ms, SpikeMon.i, '.k')
#b2.plot(SpikeMon2.t/b2.ms, SpikeMon2.i, '.g')
#b2.xlabel('Time (ms)')
#b2.ylabel('Neuron index')