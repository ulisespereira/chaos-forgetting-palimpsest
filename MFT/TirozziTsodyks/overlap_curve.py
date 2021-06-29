import numpy as np
import pickle
from classes.mft import * # solving m and then del0 and del1
import matplotlib.pyplot as plt

''' Producing  the SMFT and DMFT curves for the overlap and delta0'''
from pathlib import Path
PATH = "../../files/MFT/TirozziTsodyks/"
Path(PATH).mkdir(parents=True, exist_ok=True)

A = 5.5 #np.arange(0.2,4.2,0.2)
curves = MFTCurves()
curves.value_A(A)

	
the_alpha_S = np.arange(0.01,1.0,0.01)

# STATIC mean field theory (SMFT)
smft = []
qs = 1. #initial q
for alpha in the_alpha_S:
	del0,qs = curves.overlap_SMFT(alpha,q0=qs)
	smft.append((del0,qs))
	if qs<=1e-3:
		qs = 0.5
	print('SMFT A=',A,'|alpha=',round(alpha,2), '|qS=',round(smft[-1][1],2),'|del0=', round(smft[-1][0] * A**2,4))
smft = np.array(smft)
pickle.dump((the_alpha_S,smft),open(PATH + 'overlap_smft_A_'+str(round(A, 2))+'.p','wb'))


#DYNAMIC mean field theory (DMFT)
dmft = []
the_alpha_D = np.arange(1.,0.01,-0.01)
qd = 0.5 #initial q
x0 = np.array([0.8,0.1])
for alpha in the_alpha_D:
	x0,qd = curves.overlap_DMFT(alpha,q0=qd,x0=x0)
	dmft.append((x0[0],x0[1],qd))
	if qd<=1e-2:
		qd = 0.5
		x0 = np.array([0.8,0.1])
	elif np.abs(x0[0]-x0[1])<0.1:
		x0 = np.array([0.8,0.1])

	print('DMFT A=',A, '|alpha=', round(alpha,2) ,'|qD=', round(dmft[-1][2],2),'|del0=',round(dmft[-1][0],4),'|del1=', round(dmft[-1][1],4))

dmft = np.array(dmft)
pickle.dump((the_alpha_D,dmft),open(PATH + 'overlap_dmft_A_'+str(round(A,2))+'.p','wb'))


















