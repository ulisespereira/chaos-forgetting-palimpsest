import numpy as np
import pickle
from classes.mft import * # solving m and then del0 and del1
import matplotlib.pyplot as plt

''' Producing  the SMFT and DMFT curves for the overlap and delta0'''

from pathlib import Path
PATH = "../../../files/MFT/TirozziTsodyks/"
Path(PATH).mkdir(parents=True, exist_ok=True)

A =5.5 #np.arange(0.2,4.2,0.2)
curves = MFTCurves()
curves.value_A(A)

alpha = 0.55



#DYNAMIC mean field theory
qd = 0.5 #initial q
x0 = np.array([0.8,0.1])
x0,qd = curves.overlap_DMFT(alpha,q0=qd,x0=x0)
print('A=', A, 'alpha=',alpha,'qD=',qd,'del0=',x0[0],'del1=',x0[1])
#del_t_0 = curves.autocov_curve(alpha,x0[0],x0[0]-5e-4, qd)
del_t_1 = curves.autocov_curve(alpha,x0[0],x0[1]+1e-20, qd)

pickle.dump(del_t_1, open(PATH+'autocov_'+'_A_'+str(round(A,2))+'_alpha_'+str(alpha)+'.p','wb'))















