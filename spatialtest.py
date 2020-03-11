# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:50:29 2019

@author: lmun373
"""

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

simulation_time = 100     # Total simulation time 

###############################################################################
########                     Neuron Equations                           #######
###############################################################################
eqs = '''
dv/dt = -v/(10*ms) : volt
x : 1
y : 1
'''
###############################################################################
########                      Create Neurons                            #######
###############################################################################
def simulate():
#initialise neurons
    length = 600
    Input = b2.NeuronGroup(1, eqs, threshold='v>0.1*mV', reset='v = 0*mV', method='exact')
    Input.x = [300]
    Input.y = [300]
    
    num = 100 #Must be square number
    G = b2.NeuronGroup(num, eqs, threshold='v>0.1*mV', reset='v = 0*mV', method='exact')
    x =[]
    y = []
    for i in range(num):
        x.append(length/(np.sqrt(num)-1) * np.floor(i%(np.sqrt(num))))
        y.append(length/(np.sqrt(num)-1) * np.floor(i/np.sqrt(num)))
    
#    print("x = ", x)
#    print("y = ", y)
    
    G.x = x
    G.y = y

    Spike = b2.SpikeGeneratorGroup(1, [0], [50]*ms) 

###############################################################################
########                          Synapses                              #######
###############################################################################
    S_input = b2.Synapses(Spike, Input, on_pre='v_post += 0.2*mV') 
    S_input.connect('i==j')

    S = b2.Synapses(Input, G, on_pre='v_post += 0.2*mV') 
    S.connect('i != j',
                     p='1.0 * exp(-((x_pre-x_post)**2 + (y_pre-y_post)**2)/(2*(60)**2))')
###############################################################################
########                         Monitors                               #######
###############################################################################
    #Monitoring membrane potentials
    M = b2.StateMonitor(G, 'v', record=True)
    spikemon = b2.SpikeMonitor(G, variables = ['v', 't'])
    
    net = b2.Network(Input, G, Spike, S_input, S, spikemon)
    
    net.run(simulation_time*ms)
    
    return M, spikemon, Input, G, S
    
#b2.store()
###############################################################################
########                         Run Model                              #######
###############################################################################
spike_counts = []
for trial in range(100):
    #b2.restore()  # Restore the initial state
    M, spikemon, Input, G, S = simulate()
    # store the results
    spike_counts.append(spikemon.i)

###############################################################################
########                       Plot Graphs                              #######
###############################################################################
plot_array = []
plt.figure()
plot_array = np.concatenate(spike_counts)

plt.hist(plot_array, bins = 25)

#fig, ax = plt.subplots(2,1, figsize=(12,10), sharex=True)
#
#ax[0].plot(M.t/ms, M.v[0], label='Neuron 0')
#ax[0].set_ylabel('v')
#ax[0].legend();
#
#ax[1].plot(spikemon.t[:]/b2.ms, spikemon.i[:], '.k')
#ax[1].set_ylabel('Neuron')
#ax[1].set_xlabel('Time (ms)')

plt.figure()
plt.plot(Input.x, Input.y, 'vb')
index = np.unique(spikemon.i)
plt.plot(G.x, G.y,'or', alpha = 0.2)
    
plt.ylabel('Spatial Plot')

plt.plot(G.x[S.j[:]],G.y[S.j[:]], 'k.')

######## Checkerboard
#num = 25
#numcol = 128  #8, 32, 128
#
#X_1 = []
#Y_1 = []
#X_2 = []
#Y_2 = []
#
#Xlength = 300
#Ylength = 300
#
#space = 50
#
#dim = (np.sqrt(numcol*2))
#print(dim)
#
## Normal
#for i in range(numcol):
#    for n in range(num):
#        X_1.append(Xlength/4 * (n%5) + 50*i%(dim) + (i%(dim)) * (Xlength+space))
#        Y_1.append(Ylength/4 * np.floor(n/5) + 350*(i%(2)) + np.floor(i/dim) * 2 * (Xlength+space))
#
## Invert
#for i in range(numcol):
#    if (i%2) == 0:
#        shift = 350
#    else:
#        shift = - 350
#    for n in range(num):
#        X_2.append(Xlength/4 * (n%5) + 50*i%(dim) + (i%(dim)) * (Xlength+space))
#        Y_2.append(Ylength/4 * np.floor(n/5) + 350*(i%(2)) + np.floor(i/dim) * 2 * (Xlength+space)+ shift)
#
#plt.plot(X_1,Y_1,'k.')
#plt.plot(X_2,Y_2,'r.')