import numpy as np
import pickle
import time
from classes.network_dynamics import *
import sys
import os



tag = int(sys.argv[1])# tag realization/seed
p = int(sys.argv[2])#pattern number
tau = float(sys.argv[3])
amp =  float(sys.argv[4])
N = int(sys.argv[5])

parameters_values = dict(
        seed = tag, #random seed
        N = N, #number of neurons
        tau = 20.,#neuron timescale
        
        tau_palimpsest = tau, #time forgetting
        n_tau_palimpsest = 6, #apprimetly infinity, here 6 taus
	
        
        lr_data = False,
        #parameters learning rule
        amp = amp,
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
        T = 4000, #simulation time
        indexes_neurons = np.array([range(100)]) #neuron dynamics saved
        )

path = '/scratch/upo201/memory_and_chaos/matrices/forgetting/'

A = str(round(parameters_values['amp'], 2))
N = str(int(parameters_values['N']/1000))
real = str(int(parameters_values['seed']))
tau = str(round(parameters_values['tau_palimpsest'], 1)) #time forgetting
name_mat = 'matrix_N_' + N + 'K_seed_'+ real +'_tau_' + tau +'_A_'+ A + '.p'
conn = pickle.load(open(path+name_mat, 'rb'))
print ('pattern retrieved =', p)
ind = (p-1, p)
# dynamics
dyn = NetworkDynamics(parameters_values, conn)
dyn.dyn = True #save firing rates 
u_init = conn.myLR.g(dyn.patterns_fr[ind[1] - 1])
sol = dyn.dynamics(u_init, ind)

path = ''#path where connecitivty matrices saved
name_sim = 'retrieval_N_' + N + 'K_seed_'+ real +'_tau_' + tau +'_A_'+ A + '_p_'+ str(p)+'.p'
if path == '':
    pass
else:
    if not os.path.exists(path):
        os.makedirs(path)
results = dict(
        parameters = parameters_values,
        dynamics  = sol)
pickle.dump(results, open(path+name_sim,'wb'))


