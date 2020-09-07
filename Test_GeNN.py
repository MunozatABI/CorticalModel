# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:43:19 2020

@author: lmun373
"""

import brian2 
from brian2 import *
import brian2genn

set_device('genn')
n = 1000
duration = 1*second
tau = 10*ms
eqs = '''
dv/dt = (v0 - v) / tau : volt (unless refractory)
v0 : volt
'''
group = NeuronGroup(n, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                    refractory=5*ms, method='exact')
group.v = 0*mV
group.v0 = '20*mV * i / (n-1)'
monitor = SpikeMonitor(group)
run(duration)