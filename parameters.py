# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:07:25 2019

@author: lmun373
"""
import brian2 as b2
from brian2 import ms, mV, um, umetre, cm, siemens, ufarad, msiemens

tau1_AMPA = 0.5*ms #0.4
tau2_AMPA = 2.4*ms #1.5
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
EK = -90*mV               # Potassium #-90
ENa = 30*mV               # Sodium #30 #60
El = -10.6 * mV          # Leak -10.6mV
gl = 0.33      # 0.33

#Constant in threshold equation
C = 0.85

#Resting Membrane potential
#RV = -60*mV

##HH Parameters
##https://brian2.readthedocs.io/en/stable/resources/tutorials/3-intro-to-brian-simulations.html
#area = 20000*umetre**2
#Cm = 1*ufarad*cm**-2 * area
#gl = 5e-5*siemens*cm**-2 * area
#El = -65*mV
#EK = -90*mV
#ENa = 50*mV
#g_na = 100*msiemens*cm**-2 * area
#g_kd = 30*msiemens*cm**-2 * area
#VT = -63*mV

##Constants from Gurleen
#area = 20000*umetre**2
#Cm = 1*ufarad*cm**-2 * area
#
##Given gPeak values
#gl = 5e-5*siemens*cm**-2 * area
#gNA = 100*msiemens*cm**-2 * area
#gK = 30*msiemens*cm**-2 * area
#
##Given current reversal potentials
#El = -65*mV  
#EK = -90*mV  
#ENa = 50*mV

###Constants from Neurodynex #https://neuronaldynamics.epfl.ch/online/index.html
## neuron parameters
#El = 10.6 * b2.mV
#EK = -12 * b2.mV
#ENa = 115 * b2.mV
#gl = 0.3 * b2.msiemens
#gK = 36 * b2.msiemens
#gNa = 120 * b2.msiemens
#Cm = 1 * b2.ufarad

#Membrane threshold
#VT = -60*mV

