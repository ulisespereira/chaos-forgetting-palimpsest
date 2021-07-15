import numpy as np
import pickle
import time
#from classes.integrals import * 
from classes.mft import * # solving m and then del0 and del1
import matplotlib.pyplot as plt
from pathlib import Path
PATH = "../../files/MFT/forgetting/"
Path(PATH).mkdir(parents=True, exist_ok=True)


a = 0
# capacity curves
the_cap_s = []
the_cap_d = []
# max capacity
the_max_s = []
the_max_d = []

# taus
the_taus = []

#kernels
the_kernels_s = []
the_kernels_d = []
x = np.linspace(0,10,1000)
# parameters
the_A = [3,4,5,10,15,20,30,50,100,1000]
taus = np.arange(0.05,3,0.005)
def critical_curves(A):
    #initial condition
    del0_s = 1.
    t_s = 0
    del0_d = 1.1
    t_d = 0
    cap_s = []
    cap_d = []
    chaos = []
    tau_x = []
    for tau in taus:
        #print(tau)
        curves = MFTCurves(A,tau,a)
        del0_s,t_s = curves.capacity_SMFT(x0 = [del0_s,t_s])
        del0_d,t_d = curves.capacity_DMFT(x0 = [del0_d,t_d])
        t_ch = curves.chaos_fixed_points()

        cap_s.append(t_s)
        cap_d.append(t_d)
        chaos.append(t_ch)

        tau_x.append(tau)
        print('A=',A,'tau=',tau)
        if t_d<0:
            break

    cap_s = np.array(cap_s)
    cap_d = np.array(cap_d)
    chaos = np.array(chaos)
    curves = dict(
            taus = tau_x,
            cap_s = cap_s,
            cap_d = cap_d,
            chaos = chaos)
    return curves

if True:
    # parameters
    the_A = [1.5, 2.,3,4, 5, 10, 20]
    #initial condition
    curves = []
    for A in the_A:
        print ('|A=',A,'|a=',a)
        critical = critical_curves(A)
        curves.append(critical)
    pickle.dump(the_A, curves, open(path + 'transitions.p', 'wb'))
