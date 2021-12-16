import numpy as np
import pickle
import time
from classes.network_dynamics import *
import sys
import os


tag = int(sys.argv[1])# tag realization/seed
p = int(sys.argv[2])#pattern number
amp =  float(sys.argv[3])
N = int(sys.argv[4])

parameters_values = dict(
        seed = tag, #random seed
        N = N, #number of neurons
        tau = 20.,#neuron timescale
        p = p,
        
        tau_palimpsest = 0.1, #time forgetting
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
        dt = 0.5, #.5#time-step
        T = 8000, #400simulation time
        indexes_neurons = np.array([range(100)]) #neuron dynamics saved
        )

#tau = str(round(parameters_values['tau_palimpsest'], 2)) #time forgetting
path = '/scratch/upo201/memory_and_chaos/matrices/Tirozzi_Tsodyks/'
N = str(int(parameters_values['N']/1000))
real = str(int(parameters_values['seed']))
p = str(int(p))
A = str(round(parameters_values['amp'], 1))
#name_mat = 'matrix_N_' + N + 'K_seed_'+ real +'_tau_' + tau + '.p'
#name_mat = 'matrix_N_' + N + 'K_p_'+str(p)+'_seed_'+ real +'.p'
name_mat = 'matrix_N_' + N + 'K_p_'+ p +'_seed_'+ real + '_A_'+ A +'.p'
conn = pickle.load(open(path+name_mat, 'rb'))
print ('pattern retrieved =', 0)
#ind = (p-1, p)
ind = (0, 1)
# dynamics
dyn = NetworkDynamics(parameters_values, conn)
dyn.all_overlaps = False #save all overlaps at all times (or not)
dyn.dyn = False #save firing rates 
init_cond = 0 # what is the initial condition used
index = ind[0] 
u_init = conn.myLR.g(dyn.patterns_fr[index])

sol = dyn.dynamics(u_init, ind)

path = ''#path where connecitivty matrices saved
name_sim = 'retrieval_N_' + N + 'K_seed_'+ real +'_p_'+ str(p)+'_A_' + A +'.p'
#name_sim = 'retrieval_N_' + N + 'K_seed_'+ real +'_p_'+ str(p)+'.p'

if path == '':
    pass
else:
    if not os.path.exists(path):
        os.makedirs(path)
results = dict(
        parameters = parameters_values,
        dynamics  = sol)
#print(results)
pickle.dump(results, open(path+name_sim,'wb'))


