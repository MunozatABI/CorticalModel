# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:02:24 2020

@author: lmun373
"""
###############################################################################
########                   Import Libraries                             #######
###############################################################################
import brian2 as b2
from brian2 import *
import numpy as np
import time
import function_library as fl
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D 

b2.start_scope()

start = time.time()

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

'''

#eqs += 'I_syn = (v - Erev_AMPA) * g_AMPA + ' + ' + '.join(['(v - Erev_{}) * g_{}{}'.format(Transmitters[i],Transmitters[i],i) for i in range(len(Transmitters))]) + ' : volt\n'
#eqs += 'g_AMPA : 1\n' + ''.join(['g_{}{} : 1\n'.format(Transmitters[i], i) for i in range(len(Transmitters))])

#initialise variables
initial_values = {'theta_eq': [-54, -53, -53, -54]*mV,            #resting threshold
                   'tau_theta': [1.0, 2.0, 0.5, 1.0]*ms,              #threshold time constant
                   'tau_spike': [0.48, 1.75, 0.6, 0.48]*ms,        #time constant during spike 
                   't_spike': [0.75, 2, 0.75, 0.75]*ms,          #length of time of spike
                   'tau_m': [7, 15, 13, 7]*ms,                  #membrane time constant 
                   'gNa': [0.2, 0.14, 0.14, 0.2],                 #sodium leak
                   'gK': [1.0, 1.0, 1.3, 1.0]}                      #potassium leak                

neurons = b2.NeuronGroup(4, eqs,
          threshold = 'v > theta', 
          method = 'rk4',
          refractory = 'v > theta')

neurons.set_states(initial_values)

neurons.v = neurons.theta_eq #initialise resting potential

Thalamus = neurons[0:1]
MPE = neurons[1:2]
MP5 = neurons[2:3]
MPI = neurons[3:4]

#Spiking Input Neurons
Spike1 = b2.SpikeGeneratorGroup(1, [0], [500]*ms) 

#Spiking Input Neurons
#Spike2 = b2.SpikeGeneratorGroup(1, [0], [510]*ms) 

###############################################################################
########                          Synapses                              #######
###############################################################################

# Input_syn1 = b2.Synapses(Spike1, Thalamus, '''
# w : 1
# ''', on_pre='x_AMPA_post += w')    
# Input_syn1.connect(p = 1)
# Input_syn1.w = 1

# Input_syn2 = b2.Synapses(Spike1, MPE, '''
# w : 1
# ''', on_pre='x_AMPA_post += w')    
# Input_syn2.connect(p = 1)
# Input_syn2.w =1

# Syn1 = b2.Synapses(Thalamus, MP5, '''
# w : 1
# ''', on_pre='x_AMPA_post += w')    
# Syn1.connect(p = 1)
# Syn1.w = 0.3

# Syn2 = b2.Synapses(MPE, MP5, '''
# w : 1
# ''', on_pre='x_AMPA_post += w')    
# Syn2.connect(p = 1)
# Syn2.w = 0.3

synapses_group = []
#synapses_group.append(Input_syn1)

eqs_syn= '''
    w : 1
    '''
Inputs = [Spike1, Spike1, Thalamus]
Targets = [Thalamus, MPE, MP5]
Weights = [1, 1, 0.3]
Transmitters = ['AMPA', 'AMPA', 'AMPA']

##Putting Synapses into a list structure
for i in range(len(Inputs)):
    syn = b2.Synapses(Inputs[i], Targets[i], eqs_syn.format(tr = Transmitters[i],st = i), on_pre='x_{}_post += w'.format(Transmitters[i]))
    syn.connect(p=1)
    syn.w = Weights[i]
    synapses_group.append(syn)

###############################################################################
########                         Monitors                               #######
###############################################################################
S1 = b2.SpikeMonitor(neurons, variables='v')
M_test = b2.StateMonitor(neurons, ['v'], record = True)
M1 = b2.StateMonitor(Thalamus, ['v', 'theta'], record=True)
M2 = b2.StateMonitor(MPE, ['v', 'theta'], record=True)
M3 = b2.StateMonitor(MP5, ['v', 'theta'], record=True)
M4 = b2.StateMonitor(MPI, ['v', 'theta'], record=True)


###############################################################################
########                         Run Model                              #######
###############################################################################
net = b2.Network(b2.collect())  #Automatically add visible objects 
net.add(synapses_group)           #Manually add list of synapses
net.run(simulation_time*b2.ms)

stop = time.time()

print('Time taken:', stop - start )
print('Max V in Tha:', np.max(M1.v[0][4000:6000]))
print('Max V in MPE:', np.max(M2.v[0][4000:6000]))
print('Max V in MP5:', np.max(M3.v[0][4000:6000]))
print('Max V in MPI:', np.max(M4.v[0][4000:6000]))

for k in range(len(S1.i)):
    print('Spike in neuron', S1.i[k], 'at time', S1.t[k])

# plt.figure()
# plt.plot(M1.t[4800:5500]/b2.ms, M1.v[0][4800:5500], 'C0-', label='THA')
# plt.plot(M1.t[4800:5500]/b2.ms, M1.theta[0][4800:5500], 'C1.', label='theta')
# plt.plot(S1.t/b2.ms, S1.i, 'ob')

#####Plotting 3 neurons
fig, ax = plt.subplots(4, 1, figsize=(12,16), sharex = True)

ax[0].plot(M1.t[4000:6000]/b2.ms, M1.v[0][4000:6000], 'C0-', label='THA')
ax[0].plot(M1.t[4000:6000]/b2.ms, M1.theta[0][4000:6000], 'C1.', label='theta')
ax[1].plot(M2.t[4000:6000]/b2.ms, M2.v[0][4000:6000], 'C2-', label='MPE')
ax[1].plot(M2.t[4000:6000]/b2.ms, M2.theta[0][4000:6000], 'C1.', label='theta')
ax[2].plot(M3.t[4000:6000]/b2.ms, M3.v[0][4000:6000], 'C3-', label='MP5')
ax[2].plot(M3.t[4000:6000]/b2.ms, M3.theta[0][4000:6000], 'C1.', label='theta')
ax[3].plot(M4.t[4000:6000]/b2.ms, M4.v[0][4000:6000], 'C3-', label='MPI')
ax[3].plot(M4.t[4000:6000]/b2.ms, M4.theta[0][4000:6000], 'C1.', label='theta')
ax[3].set_xlabel('Time (ms)')
ax[3].set_ylabel('v')
plt.legend();