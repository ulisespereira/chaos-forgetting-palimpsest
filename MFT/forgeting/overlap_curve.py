import numpy as np
import pickle
import time
from classes.integrals import * 
from classes.mft import * # solving m and then del0 and del1
import matplotlib.pyplot as plt
from pathlib import Path
PATH = "../../files/MFT/forgetting/"
Path(PATH).mkdir(parents=True, exist_ok=True)


A = 3.
tau = 0.2
time = np.arange(0,0.6,0.01)
a = 0

name_file = 'overlap_vs_load_A_' + str(round(A,2)) + '_tau_'+str(round(tau,2))+'.p'

ov_d = []
ov_s = []
curves = MFTCurves(A,tau,a)
for t in time:
	del0_d,del1_d,m_d = curves.overlap_DMFT(t)
	del0_s,m_s = curves.overlap_SMFT(t)
	ov_d.append((m_d,del0_d,del1_d))
	ov_s.append((m_s,del0_s))
	print ('t=',t,'|del0_d=',del0_d,'|del1_d=',del1_d,'|m_d=',m_d,'|del0_s=',del0_s,'|m_s=',m_s)
pickle.dump((time,ov_s,ov_d),open(PATH+name_file,'wb'))


#x = curves.capacity_DMFT(tau)
