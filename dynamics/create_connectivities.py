import numpy as np
import pickle
import time
from classes.network_dynamics import *
import sys
import os

tag = int(sys.argv[1])# tag realization/seed


parameters_values = dict(
        seed = tag, #random seed
        N = 20000, #number of neurons
        tau = 20.,#neuron timescale
        
        tau_palimpsest = .5, #time forgetting
        n_tau_palimpsest = 6, #apprimetly infinity, here 6 taus
	
        #parameters learning rule
        amp = 10,
        qf = 0.5,
        bf = 1e6,
        bg = 1e6,
        xf = 0,
        xg = 0,

        #transfer function
        r_max = 2.,
        q_tanh = 0.5,
        beta = 1,
        h0 = 0,
        which_tf = 'tanh',

        #simulation
        dt = 0.5, #time-step
        T = 2500, #simulation time
        indexes_neurons = np.array([0]) #neuron dynamics saved
        )

path = 'matrices/'#path where connecitivty matrices saved
N = str(int(parameters_values['N']/1000))
real = str(int(parameters_values['seed']))
name = 'matrix_N_' + N + 'K_seed_'+ real +'.p'
if path == '':
    pass
else:
    if not os.path.exists(path):
        os.makedirs(path)
#building connectivity
conn = ConnectivityMatrix(parameters_values)
conn.connectivity_generalized_hebbian()
pickle.dump(conn, open(path+name, 'wb'))

