import numpy as np
import pickle
from classes.mft import * # solving m and then del0 and del1



from pathlib import Path
PATH = "../../files/MFT/TirozziTsodyks/"
Path(PATH).mkdir(parents=True, exist_ok=True)



myA = np.linspace(9.,0.5,200)


#chaos background
trans_bckg=[]

#chaos fixed points
trans_fxd=[]
m_fxd = 1.

#capacity static MFT
cap_SMFT = []


#capacity dynamic MFT
cap_DMFT = []



if True:
	for A in myA:
		print ('%%%%%%%%%%%%%%%%%% A=',A,'%%%%%%%%%%%%%%')

		
		
		# TT mean field equations 
		# f2 has the 0.5 multiplying for this 2**2

		curves = MFTCurvesBifurcationDiagram()
		curves.value_A(A)
		
		# capacity DMFT
		#transition chaos background
		alpha_bckg = curves.chaos_background()
		trans_bckg.append(alpha_bckg)
			
		# transition fixed points	
		del0_fxd,m_fxd,alpha_fxd = curves.chaos_fixed_points()
		trans_fxd.append(alpha_fxd)

		# capacity static mean field theory
		del0_SMFT,m_SMFT,alpha_SMFT = curves.capacity_SMFT()
		cap_SMFT.append(alpha_SMFT)
		
		alpha_DMFT = curves.capacity_DMFT()
		cap_DMFT.append(alpha_DMFT)
		
		print('alpha_c background:',alpha_bckg)
		print('alpha_c fxd points:',alpha_fxd)
		print('alpha_c SMFT:', alpha_SMFT)
		print('alpha_c DMFT:',alpha_DMFT)
	pickle.dump((myA,trans_fxd),open(PATH +'trans_fxd.p','wb'))
	pickle.dump((myA,trans_bckg),open(PATH + 'trans_bckg.p','wb'))
	pickle.dump((myA,cap_SMFT,),open(PATH + 'cap_SMFT.p','wb'))
	pickle.dump((myA,cap_DMFT,),open(PATH + 'cap_DMFT.p','wb'))
#
