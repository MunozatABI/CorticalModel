# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:46:56 2020

@author: lmun373
"""

# from pypet import Environment, cartesian_product

# #Simulation function
# def multiply(traj):
#     """Example of a sophisticated simulation that involves multiplying two values.

#     :param traj:

#         Trajectory containing
#         the parameters in a particular combination,
#         it also serves as a container for results.

#     """
#     z = traj.x * traj.y
#     traj.f_add_result('z',z, comment='I am the product of two values!')


# # Create an environment that handles running our simulation
# env = Environment(trajectory='Multiplication',filename='./HDF/example_01.hdf5',
#                   file_title='Example_01',
#                   comment='I am a simple example!',
#                   large_overview_tables=True)

# # Get the trajectory from the environment
# traj = env.trajectory

# # Add both parameters
# traj.f_add_parameter('x', 1.0, comment='Im the first dimension!')
# traj.f_add_parameter('y', 1.0, comment='Im the second dimension!')

# # Explore the parameters with a cartesian product
# traj.f_explore(cartesian_product({'x':[1.0,2.0,3.0,4.0], 'y':[6.0,7.0,8.0]}))

# # Run the simulation with all parameter combinations
# env.run(multiply)

# # Finally disable logging and close all log-files
# env.disable_logging()


#### Loading Data ####
from pypet import Trajectory

# So, first let's create a new empty trajectory and pass it the path and name of the HDF5 file.
traj = Trajectory(filename='./HDF/example_01.hdf5')

# Now we want to load all stored data.
traj.f_load(index=-1, load_parameters=2, load_results=2)

# Finally we want to print a result of a particular run.
# Let's take the second run named `run_00000001` (Note that counting starts at 0!).
print('The result of run_00000003 is: ')
print(traj.run_00000003.z)