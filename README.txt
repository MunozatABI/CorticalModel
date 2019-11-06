EsserModel is a model of TMS on the motor cortex based on (Esser, et al., 2015) 
~ the function_library.py file contains all the functions used to define neurons, synapses, equations and plotting
~ the parameters.py file contains all the parameters used in the model

STEP: Run Simple_Model.py file to see the membrane voltage activity of a single spiking neuron

-2 thalamic input neurons are defined which spike at 500*ms, as deifned by Spike1 and Spike2
-3 different types of neurons are included, these can be changed under neuron_type = []
	MPE (Excitatory Motor Coretx), MPI (Inhibitory Motor Cortex), L5E (Layer 5 Excitatory Neuron)
-4 different transmitters are included, these can be changed under Transmitters = []
	AMPA, NMDA, GABAA, GABAB
-You can define Inputs and Targets in the synapses sections, as well as the probability of connection, weight and delay
	Make sure arrays are the same size


STEP: Run column_esser.py file to see the membrane voltage and spiking activity of a column of 225 neurons and of 25 thalamic neurons

-Synapse definitions are read from the Esser_table1.csv file
-The input activity is from 37 neurons (25 Thalamic - firing at 10-20 Hz, 12 SI/PM - firing at 1 Hz)