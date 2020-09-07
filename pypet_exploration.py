# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 15:49:15 2020

@author: lmun373
"""

import pypet
import brian2
import os
import numpy as np
import Simple_Model
import function_library as fl

from pypet.environment import Environment
from pypet.brian2.network import NetworkManager

brian2.prefs.codegen.target = 'numpy'

def my_pypet_wrapper(traj):
    error = Simple_Model.objective(traj.x)
    traj.f_add_result('error', error, comment='test error from simple neuron model')


def main():
    #Set up Pypet Environment
    env = Environment(trajectory = 'simple',
                      filename = os.path.join('hdf5', 'Network.hdf5'),
                      add_time = True,
                      comment = 'Testing'
                      )
    traj = env.traj
    
    traj.f_add_parameter('x', [1])
    
    traj.f_explore({'x': [[0.2], [0.5], [0.8]]})
    
    env.run(my_pypet_wrapper)
    
    traj.f_load(filename = './hdf5/Network.hdf5', load_results=2)
    
    #traj.v_auto_load = True
    
    result = list(traj.f_get_from_runs(name='error', fast_access=True).values())
    
    return result
    

if __name__ == '__main__':
    results = main()


