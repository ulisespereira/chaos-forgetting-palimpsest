import numpy as np
import pickle
from classes.integrals import * 
from classes.mft import * # solving m and then del0 and del1
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import brentq
from pathlib import Path
PATH = "../../files/MFT/forgetting/"
Path(PATH).mkdir(parents=True, exist_ok=True)


# parameters
A = 10.
a = 0.

# transitions
t_cap_s = []
t_cap_d = []
t_chaos = []



#transition to chaos background
ker=lambda t,tau: np.exp(-(2 * t)/tau) * (t + 1)**(2 * a)
def gam(tau): 
	sol,err  = integrate.quad(ker,0,np.inf,args = (tau,))
	return sol * A**2 - 1.

tau_c_bckg = brentq(gam,0.001,2000.)
taus = np.arange(2.,0.0,-0.01)
# simulations
x0_d = [0.5,0]
x0_s = [0.5,0]
for tau in taus:
	curves = MFTCurves(A,tau,a)
	del0_s,t_s = curves.capacity_SMFT(x0=x0_s)
	del0_d,t_d = curves.capacity_DMFT(x0=x0_d)
	t_ch = curves.chaos_fixed_points()
	t_cap_s.append(t_s)
	t_cap_d.append(t_d)
	t_chaos.append(t_ch)
	x0_s = [del0_s,t_s]
	x0_d = [del0_d,t_d]
	print('tau=',tau,'|t_chaos=',t_ch,'|t_cap_d=',t_d,'|t_cap_s=',t_s,'|del0_s',del0_s,'|del0_d',del0_d)

## storing 
pickle.dump((taus,t_chaos),open(PATH + 'trans_fxd_A_' + str(round(A,2)) + '.p','wb'))
pickle.dump((taus,t_cap_s,),open(PATH + 'cap_SMFT_A_' + str(round(A,2)) + '.p','wb'))
pickle.dump((taus,t_cap_d,),open(PATH + 'cap_DMFT_A_' + str(round(A,2)) + '.p','wb'))
