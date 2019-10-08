# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:01:07 2019

@author: lmun373
"""
import numpy as np
import brian2 as b2

###############################################################################
########                    Function Definitions                       ########
########                                                               ########        
###############################################################################

def parameters():
    #Channel Parameters
    tau1_AMPA = 0.5*b2.ms
    tau2_AMPA = 2.4*b2.ms
    Erev_AMPA = 0*b2.mV
    gpeak_AMPA = 0.1
    
    tau1_GABAA = 1*b2.ms
    tau2_GABAA = 7*b2.ms
    Erev_GABAA = -70*b2.mV
    gpeak_GABAA = 0.33
    
    tau1_GABAB = 60*b2.ms
    tau2_GABAB = 200*b2.ms
    Erev_GABAB = -90*b2.mV
    gpeak_GABAB = 0.0132
    
    tau1_NMDA = 4*b2.ms
    tau2_NMDA = 40*b2.ms
    Erev_NMDA = 0*b2.mV
    gpeak_NMDA = 0.1
    
    #Constants
    EK = -90*b2.mV               # Potassium
    ENa = 30*b2.mV               # Sodium
    El = -10.6 *b2.mV
    gl = 0.33
    
    #Constant in threshold equation
    C = 0.85
    

#Function to generate randomly firing neurons for Premotor (PM) and Somatosensory
#(SI) neurons
    #num - number of randomly firing neurons
    #meanfiring - average firing rate in Hz
    #name - name of neuron group
def random_firing (num, meanfiring, name):
    eqs = f"dv/dt = (xi*sqrt(second) + {meanfiring})*Hz : 1"
    name = b2.NeuronGroup(num, eqs, threshold='v>1', reset='v=0',
                          method='euler')
    return (name)


#Function to initialise neuron parameters
    #ntype - an array of the different types of neurons 
    #        MPE: Priamry Motor Cortex Excitatory
    #        L5E: Primary Motor Cortex Layer 5 Excitatory
    #        MPI: Primary Motor Cortex Inhibitory
    #num - an array of the number of neurons of each corresponding ntypes
    #dict - dictionary to return initial values
def initialise_neurons (ntype, num, dict):
#Defining neuron parameters
    theta_eq = []
    tau_theta = []
    tau_spike = []
    t_spike = []
    tau_m = []
    g_Na = []
    g_K = []
    count = np.full((1, sum(num)), 0)
    
    for i in range(len(num)):
        for j in range(num[i]): 
        
            if ntype[i] == 'MPE':
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
                
            elif ntype[i] == 'MPI':
                theta_eq.append(-54)
                tau_theta.append(1.0)
                tau_spike.append(0.48)
                t_spike.append(0.75)
                tau_m.append(7)
                g_Na.append(0.2)
                g_K.append(1.0)
                
            elif ntype[i] == 'THA':
                theta_eq.append(-54)
                tau_theta.append(1.0)
                tau_spike.append(0.48)
                t_spike.append(0.75)
                tau_m.append(7)
                g_Na.append(0.2)
                g_K.append(1.0)
            
    dict["theta_eq"] = theta_eq*b2.mV
    dict["tau_theta"] = tau_theta*b2.ms
    dict["tau_spike"] = tau_spike*b2.ms
    dict["t_spike"] = t_spike*b2.ms
    dict["tau_m"] = tau_m*b2.ms
    dict["gNa"] = g_Na
    dict["gK"] = g_K
    dict["count"] = count             
    return (dict)

#Function to run file
    #filename - name of file for synapse definitions    
def run_file(filename):
    with open(filename,"r") as rnf:
        exec(rnf.read())

#Function to define synapses with summation variables
        #source - array of neuron subgroups where synapse orginates from
        #target - array of neuron subgroups where synapse ends
        #prob - array of probabilities of synapses
        #weight - weights of each synapse
        #n_syn - number of synapses
def create_synapses(source, target, prob, weight, n_syn):
    file = open("synapses.txt", "w+") 
    for i in range(n_syn):
        synapse_def = (
                f"syn{i} = Synapses({source[i]}, {target[i]}, model = ''' \n" 
                f"dg_syn/dt = ((tau_2 / tau_1) ** (tau_1 / (tau_2 - tau_1))*x-g_syn)/tau_1 : 1\n"
                f"dx/dt = (-x/tau_2) : 1\n"
                f"g{i}_post = g_syn : 1 (summed)\n"
                f"t_peak = ((tau_2*tau_1)/(tau_2 - tau_1)) * log (tau_2/tau_1)  : second\n"
                f"w : 1 ''', on_pre = 'x += w')\n"
                f"syn{i}.connect(p={prob[i]})\n"
                f"syn{i}.w = {weight[i]}\n\n")  
        file.write(synapse_def)
    run_file("synapses.txt")
    file.close()

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
           
#Function to visualise connectivity
       #S - synapse
def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')

def equation (eqs, type):

    if type == 'current':

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
    
        g_AMPAa : 1
        
        g_AMPAb : 1
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
        
        I_h = (gpeak_GABAB)  * ((1/(1 + exp((v/volt - (-75.0))/5.5))) ** n) * h * (v - ENa) : volt
        
        I_t = (gpeak_GABAB) * (1/(1 + exp((-v/volt + 59.0)/6.2)) ** n) * (1/(1 + exp((v/volt + 83.0)/4.0))) * (v - ENa) : volt
        
        I_nap = (gpeak_GABAB) * (1/(1 + exp(-v/volt + 55.7)/7.7)  ** n) * (v - ENa) : volt
        
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
        
        g_AMPAa : 1
        
        '''
        
    elif type == 'HH':
        
        eqs = """
        I_e = input_current(t,i) : amp
        
        membrane_Im = I_e + gNa*m**3*h*(ENa-vm) + \
            gl*(El-vm) + gK*n**4*(EK-vm) : amp
            
        alphah = .07*exp(-.05*vm/mV)/ms    : Hz
        alpham = .1*(25*mV-vm)/(exp(2.5-.1*vm/mV)-1)/mV/ms : Hz
        alphan = .01*(10*mV-vm)/(exp(1-.1*vm/mV)-1)/mV/ms : Hz
        betah = 1./(1+exp(3.-.1*vm/mV))/ms : Hz
        betam = 4*exp(-.0556*vm/mV)/ms : Hz
        betan = .125*exp(-.0125*vm/mV)/ms : Hz
        dh/dt = alphah*(1-h)-betah*h : 1
        dm/dt = alpham*(1-m)-betam*m : 1
        dn/dt = alphan*(1-n)-betan*n : 1
        
        dvm/dt = membrane_Im/C : volt
        """
    
    return eqs