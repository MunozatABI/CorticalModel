# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:01:07 2019

@author: lmun373
"""
import parameters as pm
from parameters import * 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import brian2 as b2
import re
from brian2 import ms, mV, um, NeuronGroup

###############################################################################
########                    Function Definitions                       ########      
###############################################################################

#Group of functions to generate neurons and synapses
class generate:
    
    #initialise_neuron function to initialise neuron parameters
    #ntype - an array of the different types of neurons 
    #        MPE: Priamry Motor Cortex Excitatory
    #        L5E: Primary Motor Cortex Layer 5 Excitatory
    #        MPI: Primary Motor Cortex Inhibitory
    #        THA: Thalamus
    #num - an array of the number of neurons of each corresponding ntypes
    #numcol - number of columns
    #dict - dictionary to return initial values
    def initialise_neurons (ntype, num, numcol, dict, pref):
    #Defining neuron parameters
        theta_eq = []
        tau_theta = []
        tau_spike = []
        t_spike = []
        tau_m = []
        g_Na = []
        g_K = []
        X = []
        Y = []
        Z = []
 #       count = np.full((1, sum(num)), 0)
 
        #Cortex neurons arranged in columns 
        for a in range(len(ntype)):        
             if 'L' in ntype[a]:
                 num = [50, 25, 50, 25, 50, 25] 
        
    #Define Spatial Arrangement
        Xlength = 300
        Ylength = 300
        space = 50
        
        dim = np.sqrt(numcol*2)
        
        for i in range(len(num)):
            
            for j in range(numcol): 
                
                for k in range(num[i]):
                
                    if '3' or '5' or '6' in ntype[i]:
                        n = k
                        if k > 24:
                            n = k - 25
                        
                        if pref == 0:
                            X.append(Xlength/4 * (n%5) + 50*j%(dim) + (j%(dim)) * (Xlength+space))
                            Y.append(Ylength/4 * np.floor(n/5) + 350*(j%(2)) + np.floor(j/dim) * 2 * (Xlength+space))
                        
                        elif pref == 180:
                                
                            if (j%2) == 0:
                                shift = 350
                            else:
                                shift = - 350
                                
                            X.append(Xlength/4 * (n%5) + 50*j%(dim) + (j%(dim)) * (Xlength+space))
                            Y.append(Ylength/4 * np.floor(n/5) + 350*(j%(2)) + np.floor(j/dim) * 2 * (Xlength+space)+ shift)
                        
                    else:
                        X.append(Xlength/np.sqrt(num[i]) * (n%(np.sqrt(num[i])))) 
                        Y.append(Ylength/np.sqrt(num[i]) * np.floor(n/np.sqrt(num[i])))
                        
                    if ntype[i].find('3') == 1:
                        Z.append(225)
                        
                    elif ntype[i].find('5') == 1:
                        Z.append(150)
                            
                    elif ntype[i].find('6') == 1:
                        Z.append(75)  
                        
                    else: 
                        Z.append(0)

    #Define parameters according to type

                    if ntype[i] == 'MPE' or ntype[i] == 'L3E' or ntype[i] == 'L6E':
                        theta_eq.append(-53)
                        tau_theta.append(2.0)
                        tau_spike.append(1.75)
                        t_spike.append(2.0)
                        tau_m.append(15)
                        g_Na.append(0.14)
                        g_K.append(1.0)
                        
                    elif ntype[i] == 'L5E':
                        theta_eq.append(-53)
                        tau_theta.append(0.5)
                        tau_spike.append(0.6)
                        t_spike.append(0.75)
                        tau_m.append(13)
                        g_Na.append(0.14)
                        g_K.append(1.3)
                        
                    elif ntype[i] == 'MPI' or ntype[i] == 'THA' or ntype[i] == 'L3I' or ntype[i] == 'L5I' or ntype[i] == 'L6I':
                        theta_eq.append(-54)
                        tau_theta.append(1.0)
                        tau_spike.append(0.48)
                        t_spike.append(0.75)
                        tau_m.append(7)
                        g_Na.append(0.2)
                        g_K.append(1.0)
                
        dict["theta_eq"] = theta_eq*mV
        dict["tau_theta"] = tau_theta*ms
        dict["tau_spike"] = tau_spike*ms
        dict["t_spike"] = t_spike*ms
        dict["tau_m"] = tau_m*ms
        dict["gNa"] = g_Na
        dict["gK"] = g_K
        dict['X'] = X*um
        dict['Y'] = Y*um
        dict['Z'] = Z*um      
    
        return (dict)
    
    #neuron function generates neurons
    #num - array of number of neurons according to type
    #ntype - neuron type
    #eqs - governing equations of neuron dynamics
    #numcol - number of columns
    def neurons(num, ntype, eqs, numcol, pref):
        n = sum(num)
        initial_values = {}
        generate.initialise_neurons(ntype, num, numcol, initial_values, pref)
        neurons = b2.NeuronGroup(n, eqs,
                  threshold = 'v > theta', 
                  reset = 'v = -65*mV; theta = -53*mV',
#                 events={'on_spike': 'v > theta'},
                  method = 'rk4',
                  refractory = 2*ms)
        neurons.set_states(initial_values)
        neurons.v = -65*mV
        #neurons.v = neurons.theta_eq #initialise resting potential
#        neurons.run_on_event('on_spike', 'count = count + 1')
        return neurons
    
    #column function generates a column of 225 neurons
    #numcol - number of columns
    #eqs - governing equations of neuron dynamics
    def column(numcol, eqs, pref):                                                          
        ntype = ['L3E', 'L3I', 'L5E', 'L5I', 'L6E', 'L6I']                 # Types of neurons
        num = [50, 25, 50, 25, 50, 25]                                # Number of each type of neuron in each column
        new_num = [i*numcol for i in num]
        columns = generate.neurons(new_num, ntype, eqs, numcol, pref)
            
        return columns

    #synapses function generates multiple synapses
    #Inputs - array of neuron groups
    #Targets - array of neuron groups
    #Transmitters - array of transmitters
    #p - probabilities
    #w - weights
    #delay - delay
    def synapses(Inputs, Targets, Transmitters, prob, w, delay, S = None):
        synapses_group = []
        eqs_syn= equation('synapse')

        for i in range(len(Inputs)):
            
            if S is not None:
                st = i
            else:
                st = chr(97 + i)
                
            syn = b2.Synapses(Inputs[i], Targets[i], eqs_syn.format(tr = Transmitters[i],st = st), on_pre='x_{}{} += w'.format(Transmitters[i], st))
            #syn.connect(j = 'i', p=prob[i])
            syn.connect(p=prob[i])#j = 'i'
            #radius = 2
            #syn.connect(condition = 'i != j', p='{} * exp(-((X_pre-X_post)**2 + (Y_pre-Y_post)**2 + (Z_pre-Z_post)**2)/(2*(1*{})**2))'.format(prob[i], radius))
            syn.w = w[i]
            syn.delay = delay[i]*ms
            synapses_group.append(syn)
            
        return synapses_group    
    
    #model_synapses function generates synapses for column_esser model
    #table - table of synapse details from Esser, 2005
    #neuron_group - column of neurons representing M1, PM, SM, Thalamus
    def model_synapses(table, neuron_group):
        all_synapses=[]
        src_group=[]
        tgt_group=[]
        eqs_syn= equation('synapse')
        for i, r in table.iterrows():
            src = r.loc['SourceLayer'] + re.sub('[()]', '', r.loc['SourceCellType'])
            tgt = r.loc['TargetLayer'] + re.sub('[()]', '', r.loc['TargetCellType'])
            syn = b2.Synapses(neuron_group[src], neuron_group[tgt],
                              model = eqs_syn.format(tr = r.loc['Transmitter'],st = i),
                              method = 'rk4',
                              on_pre='x_{}{} += w'.format(r.loc['Transmitter'], i))
            syn.connect(condition = 'i != j', p='{} * exp(-((X_pre-X_post)**2 + (Y_pre-Y_post)**2)/(2*(37.5*{})**2))'.format(r.loc['Pmax'],r.loc['Radius'])) #Gaussian connectivity profile
            syn.w = (r.loc['Strength']/2)  #Weights scaled to match Iriki et al., 1991
            syn.delay = r.loc['MeanDelay']*ms
            #post.delay = '{}*ms'.format(,r.loc['VCond'])
            #syn.delay = ('{}*ms + (((X_pre-X_post)**2 + (Y_pre-Y_post)**2 + (Z_pre-Z_post)**2)/({}*(b2.metre/b2.second)))'.format(r.loc['MeanDelay'],r.loc['VCond']))
            all_synapses.append(syn)
            src_group.append(src)
            tgt_group.append(tgt)
        return src_group, tgt_group, all_synapses
    
    #spikegen function generates spikes
    #num - number of spiking neurons
    #indices - array of neuron indicies to spike, corresponding to times
    #times - array of times of spikes
    def spikes(num1, num2, duration):
        
        times = []
        indices = []
        
        #Thalamus 10-20Hz firing
        for j in range(num1):
            x = 0
            numspikes = np.random.randint(10*(duration/ms/1000), 20*(duration/ms/1000), 1)
#            numspikes = [1]
            s1 = np.random.uniform(50, 100, numspikes[0])
#            s1 = ([150])
            #spikes = np.ones(numspikes) * 50
            #add in asynchrony
            if (j%2==1):
                    s1[0] = s1[0] + 1000/numspikes[0]
            times.extend(list(np.round(np.cumsum(s1), 1)))
            indices.extend(list(np.ones_like(s1) * j))
            
        #mu, sigma = round(1000/1), round(100/1.) 
        
        num_spikes2 = round(1*duration/ms/1000)   
        #SI and PM 1Hz firing
        for j in range(num2):
            x = 0
#           s2 = np.random.normal(mu, sigma, num_spikes2)
            s2 = np.random.uniform(1, 1000, num_spikes2)
            for k in range(len(s2)):
                x += s2[k]
                times.append(round(x))
                indices.append(j + num1)
        
        input_indices= b2.array(indices)
        input_times = times*ms
        Spikes = b2.SpikeGeneratorGroup(num1+num2, input_indices, input_times)
        return Spikes
                
#Function to define multiple synapses
        #sources - array of neuron subgroups
        #targets - array of neuron subgroups
        #dict - dict for synapses
def input_firing(mean_frequency, times, zeros, simulation_time): 
    
    mu, sigma = round(1000/mean_frequency), round(100/mean_frequency) 
    num_spikes = round(simulation_time/b2.ms/mean_frequency)
    s = np.random.normal(mu, sigma, num_spikes)
    x = 0
    
    for i in range(len(s)):
        x += s[i]
        times.append(np.round(x))
        zeros.append(0)
        
    return times, zeros

def calculate_isi(spike_times, neuron_index):
    isis = []
    neurons = np.unique(neuron_index)
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    
    for i in range(len(neurons)):
        idxs = get_indexes(neurons[i],neuron_index)
        intime = [spike_times[x] for x in idxs]
        isis += [np.diff(intime)]
        
    return isis

class visualise():

    #Function to plot the membrane voltage of cells
    #labels - graph titles
    #monitors - state monitors which contatin t and v information
    #num - index number of neuron to get data from
    def membrane_voltage(labels, monitors, num):
        b2.figure(figsize=(12,4))
        for i in range(len(labels)):
            b2.plot(monitors[i].t/b2.ms, monitors[i].v[num[i]], 'C{}'.format(i), label=labels[i])
            b2.xlabel('Time (ms)')
            b2.ylabel('v')
            b2.legend();        
            
    #Function to plot average firing of cells
    def average_firing(labels, monitors, simulation_time):
        b2.figure(figsize=(12,4))
        for i in range(len(labels)):
            uniqueValues_input, occurCount_input = np.unique(monitors[i].i, return_counts=True)
            frequencies_input = occurCount_input/((simulation_time)/1000)
            plt.gca().set_title(labels[i])
            plt.hist(frequencies_input,bins = 30)
    
    #Function to visualise synapse connectivity in 1D
           #S - synapses
           #source_subidx - array of offsets for subgrouping neurons
           #target_subidx - array of offsets for subgrouping neurons
    def connectivity(S, source_subidx, target_subidx):
        Ns = []
        Nt = []

        for n in range(len(S)):
            Ns.append(len(S[n].i))
            Nt.append(len(S[n].j))
        b2.figure(figsize=(10, 4))
        b2.subplot(121)
        b2.plot(b2.zeros(sum(Ns)), b2.arange(sum(Ns)), 'ok', ms=5)
        b2.plot(b2.ones(sum(Nt)), b2.arange(sum(Nt)), 'ok', ms=5)
        for n in range(len(S)):
            for i, j in zip(S[n].i, S[n].j):
                b2.plot([0, 1], [i+source_subidx[n], j+target_subidx[n]], '-k', alpha = 0.2)
        b2.xticks([0, 1], ['Source', 'Target'])
        b2.ylabel('Neuron index')
        b2.xlim(-0.1, 1.1)
        b2.ylim(-1, np.max(S[len(S)-1].j)+max(target_subidx)+3)
        b2.subplot(122)
        for n in range(len(S)):
            b2.plot((S[n].i)+source_subidx[n], (S[n].j)+target_subidx[n], 'ok')
        b2.xlim(-1, np.max(S[len(S)-1].i)+max(source_subidx)+1)
        b2.ylim(-1, np.max(S[len(S)-1].j)+max(target_subidx)+3)
        b2.xlabel('Source neuron index')
        b2.ylabel('Target neuron index')
        
        #Function to visualise synapse connectivity in 3D
           #S - synapses
           #N - neurons
           #source_subidx - array of offsets for subgrouping neurons
           #target_subidx - array of offsets for subgrouping neurons
    def spatial_connectivity(S, N, source_subidx, target_subidx):
        x_src = []
        y_src = []
        z_src = []
        x_tgt = []
        y_tgt = []
        z_tgt = []
        
        for m in range(len(S)):
            
            for n in range(len(S[m].i)):
                x_src.append(N[S[m].i[n]].X)
                y_src.append(N[S[m].i[n]].Y)
                z_src.append(N[S[m].i[n]].Z)
                
            for n in range(len(S[m].j)):
                x_tgt.append(N[S[m].j[n]].X)
                y_tgt.append(N[S[m].j[n]].Y)
                z_tgt.append(N[S[m].j[n]].Z)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_src/b2.umetre, y_src/b2.umetre, z_src/b2.umetre, s = 10, color = 'r', marker ='^')
        ax.scatter(x_tgt/b2.umetre, y_tgt/b2.umetre, z_tgt/b2.umetre, s = 1, color = 'k', marker ='o')
        
        for x_o, y_o, z_o, x_t, y_t, z_t in zip(x_src, y_src, z_src, x_tgt, y_tgt, z_tgt):
            ax.plot([x_o[0]/b2.umetre, x_t[0]/b2.umetre], [y_o[0]/b2.umetre, y_t[0]/b2.umetre], [z_o[0]/b2.umetre, z_t[0]/b2.umetre], color ='blue', alpha = 0.1)
        
   #Function to visualise connectivity of individual neurons in network and calculate distances
       #neuron_num: number of input neuron
       #synapses: synapses group
       #columnsgroup: neuron group
    def single_neuron_connectivity(neuron_num, synapses, neurongroup, plot = True):
       tgt_list = []
       distances = []
       weights = []
       
       get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
           
       for n in range(len(synapses)):
            idxs = get_indexes(neuron_num, synapses[n].i)
            
            tgts = [synapses[n].j[x] for x in idxs]
            tgt_list.append(tgts)
            wgts = [synapses[n].w[x] for x in idxs]
            weights.append(wgts)
     
       flat_tgt = [val for sublist in tgt_list for val in sublist]
       flat_weight = [val for sublist in weights for val in sublist]
        
       x_ori = [neurongroup[neuron_num].X]
       y_ori = [neurongroup[neuron_num].Y]
       z_ori = [neurongroup[neuron_num].Z]
        
       x_vals = [neurongroup[k].X for k in flat_tgt]
       y_vals = [neurongroup[k].Y for k in flat_tgt]
       z_vals = [neurongroup[k].Z for k in flat_tgt]
       
       for i in range(len(flat_tgt)):
           distances.extend(np.sqrt((x_vals[i] - x_ori[0]) ** 2 + (y_vals[i] - y_ori[0]) ** 2 + (z_vals[i] - z_ori[0]) ** 2))
        
       if plot == True:
            #### 3D Connectivity Plot ####
             fig = plt.figure(figsize=(10,8))
             ax = fig.add_subplot(111, projection='3d')
             ax.scatter(x_vals/b2.umetre, y_vals/b2.umetre, z_vals/b2.umetre, s = flat_weight, color = 'k', marker='o', alpha = 1)
             ax.scatter(x_ori/b2.umetre, y_ori/b2.umetre, z_ori/b2.umetre, s = 20, color = 'red', marker='^', alpha = 1)
             for x, y, z in zip(x_vals, y_vals, z_vals):
                ax.plot([x_ori[0]/b2.umetre, x[0]/b2.umetre], [y_ori[0]/b2.umetre, y[0]/b2.umetre], [z_ori[0]/b2.umetre, z[0]/b2.umetre], color ='blue', alpha = 0.3)
             ax.set_xlabel("Distance (um)")
             ax.set_ylabel("Distance (um)")
             ax.set_zlabel("Distance (um)")
             ax.set_xlim(0, 300)
             ax.set_ylim(0, 300)
             #ax.set_zlim(0, 300)
            ##### 2D Connectivity Plot
            # plt.figure(figsize=(10,8))
            # plt.scatter(x_vals/b2.umetre, y_vals/b2.umetre,s = 10, marker='.', alpha = 1)
            # plt.scatter(x_ori/b2.umetre, y_ori/b2.umetre, s = 20, color = 'red', marker='^', alpha = 1)
            # for i, j, w in zip(x_vals, y_vals, flat_weight):
            #     plt.plot([x_ori[0]/b2.umetre, i/b2.umetre], [y_ori[0]/b2.umetre, j/b2.umetre], '-b', linewidth = w, alpha = 0.2)
            # plt.xlabel("Distance (um)")
            # plt.ylabel("Distance (um)")
       
       return distances
       

    def connectivity_distances(neurongroup, synapses):
        distancearray = []
        for neuron_num in range(len(neurongroup)):
            distancearray += visualise.single_neuron_connectivity(neuron_num, synapses, neurongroup, plot = False)

        plt.hist(distancearray, bins = 20)
        
        return distancearray
           
#Define equations
   # start_time = t * int(v>=theta) - count*dt : second
def equation (type):

    if type == 'current':
        
        eqs = '''
        dtheta/dt = (-1*(theta - theta_eq)
                     + C * (v - theta_eq)) / tau_theta
                     : volt
        
        dv/dt = ((-gNa*(v-ENa) - gK*(v-EK) - I_syn - gl*(v-El)))
            / (tau_m)   
            - int(v > theta) * int(t < (lastspike + t_spike)) * ((v - ENa) / (tau_spike))
              : volt
                      
        theta_eq : volt
        
        tau_theta : second
        
        tau_spike : second
        
        t_spike : second
        
        tau_m : second
        
        gNa : 1
        
        gK : 1
        
        X : meter
        
        Y : meter
        
        Z : meter
        '''
        
    elif type == 'test':

        eqs = '''
        dtheta/dt = (-1*(theta - theta_eq)
                     + C * (v - theta_eq)) / tau_theta
                     : volt
        
        dv/dt = ((-gNa*(v-ENa) - gK*(v-EK) - gl*(v-El) - I_syn  + I))
            / (tau_m/8)   
            - int(v >= theta) * int(t < (lastspike + t_spike)) * ((v - ENa) / (tau_spike/2))
              : volt
              
        I = ta(t) : volt
        
        theta_eq : volt
        
        tau_theta : second
        
        tau_spike : second
        
        t_spike : second
        
        tau_m : second
        
        gNa : 1
        
        gK : 1
        
        X : meter
        
        Y : meter
        
        Z : meter
        '''
        
        
          
    elif type == 'I_int':
        eqs = '''
        dtheta/dt = (-1*(theta - theta_eq)
                     + C * (v - theta_eq)) / tau_theta
                     : volt
        
        dv/dt = ((-gNa * (v-ENa)
                 - gK * (v-EK) 
                 - I_syn + I_int)
                / tau_m    
                - int(v >= theta) * int(t < (start_time + t_spike)) * (v - ENa) / tau_spike)
                  : volt
        
        start_time = t * int(v>=theta) - count*dt : second
        
        I_int = I_h + I_t + I_nap : volt
        
        I_h = 0.1 * ((1/(1 + exp((v/volt - (-75.0))/5.5))) ** n) * h * (v - ENa) : volt
        
        I_t = 0.1 * (1/(1 + exp((-v/volt + 59.0)/6.2)) ** n) * (1/(1 + exp((v/volt + 83.0)/4.0))) * (v - ENa) : volt
        
        I_nap = 0.1 * (1/(1 + exp(-v/volt + 55.7)/7.7)  ** n) * (v - ENa) : volt
        
        dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/
            ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
        
        dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
            (exp((15.*mV-v+VT)/(5.*mV))-1.)/
            ms*(1.-n)-.5*exp((10.*mV-v+VT)/
            (40.*mV))/ms*n : 1 
        
        theta_eq : volt
        
        tau_theta : second
        
        tau_spike : second
        
        t_spike : second
        
        tau_m : second
        
        gNa : 1
        
        gK : 1
        
        count : 1
        
        X : meter
        
        Y : meter
        
        Z : meter
        
        '''
        
    elif type == 'HH':
    #Model of Hodgkin Huxley neuron dynamics
    #from http://neuronaldynamics.epfl.ch
    #https://neuronaldynamics-exercises.readthedocs.io/en/latest/_modules/neurodynex/hodgkin_huxley/HH.html#getting_started
        eqs = '''
        dtheta/dt = (-1*(theta - theta_eq)
                     + C * (v - theta_eq)) / tau_theta
                     : volt
        
        dv/dt = ((gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm) + I_syn : volt
        dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
            (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
            (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
        dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
            (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
        dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1

        I : amp
        
        theta_eq : volt
        
        tau_theta : second
        
        tau_spike : second
        
        t_spike : second
        
        tau_m : second
        
        gNa : 1
        
        gK : 1
        
        count : 1
        
        X : meter
        
        Y : meter
        
        Z : meter
        '''
        
    elif type == 'gurleen':
        
        eqs = '''
            dv/dt = (gl*(El-v) - 
                     gNA*(m_NA*m_NA*m_NA)*h_NA*(v-ENa) - 
                     gK*(n_K*n_K*n_K*n_K)*(v-EK) + 
                     g_H*m_H*(v-EH) + 
                     g_NMDA*(v-ENMDA) + 
                     g_AMPA*(v-EAMPA) + 
                     g_GABAB*(v-EGABAB) +
                     I_ext)/Cm : volt (unless refractory)
            
            dm_H/dt = (m_H_inf - m_H) / (tau_m_H * ms) : 1
            m_H_inf = 1/(1 + exp((v - VT)/(5.5*mV))) : 1
            tau_m_H = 1/
                    (exp((-14.59*mV - 0.086 * v)/(1.*mV)) + 
                    exp((-1.87*mV + 0.0701 * v)/(1.*mV))) : 1
            
            dm_NA/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
                (exp((13.*mV-v+VT)/(4.*mV))-1.)/
                ms*(1-m_NA)-0.28*(mV**-1)*(v-VT-40.*mV)/
                (exp((v-VT-40.*mV)/(5.*mV))-1.)/
                ms*m_NA : 1
            
            dh_NA/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/
                ms*(1.-h_NA)-4./
                (1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h_NA : 1
            
            dn_K/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
                (exp((15.*mV-v+VT)/(5.*mV))-1.)/
                ms*(1.-n_K)-.5*exp((10.*mV-v+VT)/
                (40.*mV))/ms*n_K : 1
            
            dk/dt = (-(k - -53*mV) + 
                     0.85*(v - -53*mV))/(2 * ms) : volt 
        '''
        
    elif type == 'synapse':
    #General Model for synapse equations 
    #eqs_syn from https://brian2.readthedocs.io/en/stable/user/converting_from_integrated_form.html
    #Biexponential synapse
        eqs = '''
        dg_{tr}_syn{st}/dt = ((tau2_{tr} / tau1_{tr}) ** (tau1_{tr} / (tau2_{tr} - tau1_{tr}))*x_{tr}{st}-g_{tr}_syn{st})/tau1_{tr} : 1 (clock-driven)
        dx_{tr}{st}/dt =  (-x_{tr}{st}/tau2_{tr}) : 1 (clock-driven)
        g_{tr}{st}_post = g_{tr}_syn{st} : 1 (summed)
        w : 1
        '''
    
    elif type =='synapse2':
        eqs = '''
        dg_{tr}_syn{st}/dt = g_{tr}_syn*
            ( ( (exp(-t/tau2_{tr}))/tau2_{tr} - (exp(-t/tau1_{tr}))/tau1_{tr})/
            exp(-tpeak_{tr}/tau1_{tr}) - exp(-tpeak_{tr}/tau2_{tr}) ) : 1
        tpeak_{tr} = ((tau2_{tr}*tau1_{tr})/(tau2_{tr} - tau1_{tr}))*
                np.ln(tau2_{tr}/tau1_{tr}) : 1
        g_{tr}{st}_post = g_{tr}_syn{st} : 1 (summed)
        w : 1
        '''
        
    elif type =='simple':
        eqs = '''
        dv/dt = -v/(10*ms) : volt
        '''
    return eqs