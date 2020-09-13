# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 20:07:47 2020

@author: lmun373
"""

import logging
import os # For path names being viable under Windows and Linux
import matplotlib.pyplot as plt

from pypet import Trajectory
from pypet.environment import Environment
from pypet.brian2.parameter import Brian2Parameter, Brian2MonitorResult
from pypet.utils.explore import cartesian_product

import brian2 as b2
from brian2 import ms, mV

# We define a function to set all parameter
def add_params(traj):
    """Adds all necessary parameters to `traj`."""
    
    # We set the BrianParameter to be the standard parameter
    traj.v_standard_parameter=Brian2Parameter
    traj.v_fast_access=True

    # Add parameters we need for our network
    traj.f_add_parameter('Net.tau1_AMPA',0.5*ms)
    traj.f_add_parameter('Net.tau2_AMPA',2.4*ms)
    traj.f_add_parameter('Net.Erev_AMPA',0*mV)
    traj.f_add_parameter('Net.gpeak_AMPA',0.1)
    traj.f_add_parameter('Net.tau1_GABAA',1*ms)
    traj.f_add_parameter('Net.tau2_GABAA',7*ms)
    traj.f_add_parameter('Net.Erev_GABAA',-70*mV)
    traj.f_add_parameter('Net.gpeak_GABAA',0.33)
    traj.f_add_parameter('Net.tau1_GABAB',60*ms)
    traj.f_add_parameter('Net.tau2_GABAB',200*ms)
    traj.f_add_parameter('Net.Erev_GABAB',-90*mV)
    traj.f_add_parameter('Net.gpeak_GABAB',0.0132)
    traj.f_add_parameter('Net.tau1_NMDA',4*ms)
    traj.f_add_parameter('Net.tau2_NMDA',40*ms)
    traj.f_add_parameter('Net.Erev_NMDA',0*mV)
    traj.f_add_parameter('Net.gpeak_NMDA',0.1)
    traj.f_add_parameter('Net.EK',-90*mV)
    traj.f_add_parameter('Net.ENa',30*mV)
    traj.f_add_parameter('Net.El',-10.6*mV)
    traj.f_add_parameter('Net.gl',0.33)
    traj.f_add_parameter('Net.C',0.85)
    traj.f_add_parameter('Net.w',0.1)
    traj.f_add_parameter('Net.p',1)

# This is our job that we will execute
def run_net(traj):
    """Creates and runs BRIAN network based on the parameters in `traj`."""
    
    Transmitters = ['AMPA', 'AMPA', 'AMPA'] #### Define Transmitters
    Num_N = 4 #### Number of neurons
    
    eqs='''
    dtheta/dt = (-1*(theta - theta_eq)
                 + C * (v - theta_eq)) / tau_theta
                 : volt
    
    dv/dt = ((-gNa*(v-ENa) - gK*(v-EK) - I_syn - gl*(v-El)))
            / tau_m    
            - int(v > theta) * int(t < (lastspike + t_spike)) * ((v - ENa) / (tau_spike))
                  : volt
    
    theta_eq : volt
    tau_theta : second
    tau_spike : second
    t_spike : second
    tau_m : second
    gNa : 1
    gK : 1
    '''
    eqs += 'I_syn = (v - Erev_AMPA) * g_AMPA + ' + ' + '.join(['(v - Erev_{}) * g_{}{}'.format(Transmitters[i],Transmitters[i],i) for i in range(len(Transmitters))]) + ' : volt\n'
    eqs += 'g_AMPA : 1\n' + ''.join(['g_{}{} : 1\n'.format(Transmitters[i], i) for i in range(len(Transmitters))])
    
    eqs_syn = '''
    dg_{tr}_syn{st}/dt = ((tau2_{tr} / tau1_{tr}) ** (tau1_{tr} / (tau2_{tr} - tau1_{tr}))*x_{tr}{st}-g_{tr}_syn{st})/tau1_{tr} : 1
    dx_{tr}{st}/dt =  (-x_{tr}{st}/tau2_{tr}) : 1
    g_{tr}{st}_post = g_{tr}_syn{st} : 1 (summed)
    w : 1
    '''

    # Create a namespace dictionary
    namespace = traj.Net.f_to_dict(short_names=True, fast_access=True)
    
    # Create the Neuron Group
    neurons = b2.NeuronGroup(Num_N, eqs,
          threshold = 'v > theta', 
          reset = 'v = theta_eq',
          method = 'rk4',
          refractory = 1.4*ms,
          namespace = namespace)
    
    #initialise variables
    initial_values = {'theta_eq': [-54, -53, -53, -54]*mV,            #resting threshold
                       'tau_theta': [1.0, 2.0, 0.5, 1.0]*ms,              #threshold time constant
                       'tau_spike': [0.1, 1.2, 0.2, 0.1]*ms,        #time constant during spike ### CHANGED
                       't_spike': [0.75, 2, 0.75, 0.75]*ms,          #length of time of spike
                       'tau_m': [1, 5, 3, 1]*ms,                  #membrane time constant ### CHANGED
                       'gNa': [0.2, 0.14, 0.14, 0.2],                 #sodium leak
                       'gK': [1.0, 1.0, 1.3, 1.0]}                 #potassium leak
    
    neurons.set_states(initial_values) #initialise parameters
    
    neurons.v = neurons.theta_eq #initialise resting potential

    #Define subgroups
    Thalamus = neurons[0:1]
    MPE = neurons[1:2]
    MP5 = neurons[2:3]
    MPI = neurons[3:4]
    
    Inputs = [Thalamus, Thalamus, Thalamus] #### Define Inputs
    Targets = [MPE, MP5, MPI] #### Define Targets
    
    #Spiking Input Neurons
    Spike1 = b2.SpikeGeneratorGroup(1, [0], [500]*ms)
    
    #Generate Spike in Thalamus Neuron
    Input_syn1 = b2.Synapses(Spike1, Thalamus, '''
    dg_AMPA_syn/dt = ((tau2_AMPA / tau1_AMPA) ** (tau1_AMPA / (tau2_AMPA - tau1_AMPA))*x_AMPA-g_AMPA_syn)/tau1_AMPA : 1
    dx_AMPA/dt =  (-x_AMPA/tau2_AMPA) : 1 (clock-driven)
    g_AMPA_post = g_AMPA_syn : 1 (summed)
    w : 1
    ''', on_pre='x_AMPA += w', namespace = namespace)    
    Input_syn1.connect(p = 1)
    Input_syn1.w = 1
    
    synapses_group = []
    synapses_group.append(Input_syn1)

    ##Putting Synapses into a list structure
    for i in range(len(Inputs)):
        syn = b2.Synapses(Inputs[i], Targets[i], eqs_syn.format(tr = Transmitters[i],st = i), on_pre='x_{}{} += w'.format(Transmitters[i], i), namespace=namespace)
        syn.connect(p = traj.p)
        syn.w = traj.w
        synapses_group.append(syn)

    #Create network object
    net = b2.Network(b2.collect()) 
    net.add(synapses_group) 

    # Create a Spike Monitor
    MSpike=b2.SpikeMonitor(neurons)
    net.add(MSpike)
    # Create a State Monitor
    MStateV = b2.StateMonitor(neurons, variables=['v', 'theta'],record=True)
    net.add(MStateV)

    #Run for 1000milliseconds
    net.run(1000*ms,report='text')

    # Add the monitors to results
    traj.v_standard_result = Brian2MonitorResult
    traj.f_add_result('SpikeMonitor',MSpike)
    traj.f_add_result('StateMonitorV', MStateV)
    
def postproc(traj, filename):
    traj = Trajectory(filename=filename, dynamically_imported_classes = [Brian2MonitorResult, Brian2Parameter])  
    traj.f_load(index=-1, load_parameters=2, load_results=2)
    Spikemon=traj.results.runs.run_00000000.SpikeMonitor
    Statemon=traj.results.runs.run_00000000.StateMonitorV
    return Spikemon, Statemon

def main():
    # Let's be very verbose!
    logging.basicConfig(level = logging.INFO)

    # Let's do multiprocessing this time with a lock (which is default)
    filename = os.path.join('hdf5', 'example_49.hdf5')
    env = Environment(trajectory='Example_49_BRIAN2',
                      filename=filename,
                      file_title='Example_49_Brian2',
                      comment = 'Go Brian2!',
                      dynamically_imported_classes=[Brian2MonitorResult, Brian2Parameter]
                      )

    traj = env.trajectory

    #add the parameters
    add_params(traj)

    #explore the different parameters
    traj.f_explore(cartesian_product({traj.f_get('ENa').v_full_name:[30*mV, 60*mV],
                           traj.f_get('EK').v_full_name:[-90*mV,-80*mV]}))


    # 2nd let's run our experiment
    env.run(run_net)

    # You can take a look at the results in the hdf5 file if you want!

    # Finally disable logging and close all log-files
    env.disable_logging()
    
    SpikeMon, StateMon = postproc(traj, 'hdf5/example_49.hdf5')
    
    plt.plot(StateMon.t, StateMon.v[0])

if __name__ == '__main__':
    main()