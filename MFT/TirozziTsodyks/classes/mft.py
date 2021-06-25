import numpy as np
from scipy.optimize import brentq,root,fmin_tnc
from scipy.stats import multivariate_normal as mvnormal

class CriticalOverlapTT:
	'''This is a supper class of gaussian integrals'''

	def __init__(self,LR):	
		self.GI = Integrals(LR)
		self.m_min = 1e-2 # below this threshold m is zero
		self.max_iter = 5000

		

	def critical_point_TT(self,del0_init):
		field = lambda x:self.GI.Doverlap_TT(x)-1.
		del0_c = brentq(field,0,2000.)
		return del0_c,0.


class Integrals:
	'''This is a class where all the integrals are computed'''

	def __init__(self):
		self.A = 3.	
		
		# grid normal integration
		self.dx=0.01 #
		x_min=-10.
		x_max=10.
		self.xgrid = np.arange(x_min,x_max,self.dx) #grid on x
		self.normal_pdf =self.std_normal(self.xgrid) #standard normal pdf
	
	def std_normal(self,x):
		sigma=1. # standard normal random variable passed thorugh transfer functiion
		mu=0
		pdf=(1./np.sqrt(2 * np.pi * sigma**2))*np.exp(-(1./2.)*((x-mu)/sigma)**2)
		return pdf
	
	# Doverlap TT	
	def Doverlap(self,del0):	
		Dphi = lambda x: self.A * (1/np.cosh(self.A * x))**2
		sdfield = np.sqrt(del0) * self.xgrid
		DTF = Dphi(sdfield)
		sol = self.dx * np.einsum('i,i',DTF,self.normal_pdf)
		return sol

	# overlap TT	
	def overlap(self,del0,m):	
		phi = lambda x: np.tanh(self.A * x)
		sdfield = np.sqrt(del0) * self.xgrid
		TF = phi(m+sdfield)
		sol = self.dx * np.einsum('i,i',TF,self.normal_pdf)
		return sol

	# TF * TF TT	
	def TF(self,del0,m):	
		phi = lambda x: np.tanh(self.A * x)
		sdfield = np.sqrt(del0) * self.xgrid
		TF2 = phi(sdfield+m)**2
		sol = self.dx * np.einsum('i,i',TF2,self.normal_pdf)
		return sol
	
	# DTF * DTF  TT	
	def DTF(self,del0,m):	
		Dphi = lambda x: self.A * (1/np.cosh(self.A * x))**2
		sdfield = np.sqrt(del0) * self.xgrid
		DTF2 = Dphi(sdfield + m)**2
		sol = self.dx * np.einsum('i,i',DTF2,self.normal_pdf)
		return sol

	# TF * TF TT	
	def Int_TF2(self,del0,m):	
		int_phi = lambda x: np.log(np.cosh(self.A * x))
		sdfield = np.sqrt(del0) * self.xgrid
		INT_TF2 = int_phi(sdfield+m)**2
		sol = self.dx * np.einsum('i,i',INT_TF2,self.normal_pdf)
		return sol
	# TF * TF TT	
	def Int_TF(self,del0,m):	
		int_phi = lambda x: np.log(np.cosh(self.A * x))
		sdfield = np.sqrt(del0) * self.xgrid
		INT_TF = int_phi(sdfield+m)
		sol = self.dx * np.einsum('i,i',INT_TF,self.normal_pdf)
		return sol


	# integral IntTF**2 for del1!=del0 fast version
	def IntTF_del1(self,del0,del1,m):		
		if np.abs(del0-del1)<=0:
			return self.Int_TF2(del0,m)
		else:
			if del1<0: # del1 cannot be negative
				del1 = 0
			A = self.A
			int_phi = lambda x: np.log(np.cosh(x))
			meanfield =   m
			#pdf
			x_min=-10 * np.sqrt(del0)
			x_max=10 * np.sqrt(del0)
			dx=(x_max-x_min)/500.
			xgrid = np.arange(x_min,x_max,dx) #grid on x
			x, y = np.mgrid[x_min:x_max:dx, x_min:x_max:dx]
			self.pos = np.empty(x.shape + (2,))
			self.pos[:, :, 0] = x 
			self.pos[:, :, 1] = y
			cov = np.array([[del0,del1],[del1,del0]])
			mypdf = mvnormal([0,0],cov)
			the_pdf = mypdf.pdf(self.pos)
			#int tf
			int_tf = int_phi(A * np.add.outer(xgrid,meanfield))
			#solution
			sol=(dx**2) * np.einsum('i,ij,j',int_tf,the_pdf,int_tf)
		return sol
	
	

	# integral TF**2 for del1!=del0	fast version
	def TF_del1(self,del0,del1,m):
		if np.abs(del0-del1)<=0: # if del1 near del1 corr gauss not definted
			return self.TF(del0,m)
		else:
			if del1<0: # del1 cannot be negative
				del1 = 0
			A = self.A
			meanfield = m
			phi = lambda x: np.tanh(x)
			#pdf
			x_min = -10 * np.sqrt(del0)
			x_max = 10 * np.sqrt(del0)
			dx = (x_max-x_min)/1000.
			xgrid = np.arange(x_min,x_max,dx) #grid on x
			x, y = np.mgrid[x_min:x_max:dx, x_min:x_max:dx]
			self.pos = np.empty(x.shape + (2,))
			self.pos[:, :, 0] = x 
			self.pos[:, :, 1] = y
			cov = np.array([[del0,del1],[del1,del0]])
			mypdf = mvnormal([0,0],cov)
			the_pdf = mypdf.pdf(self.pos)
			#tf
			tf = phi(A * np.add.outer(xgrid,meanfield))
			#solution
			sol = (dx**2) * np.einsum('i,ij,j',tf,the_pdf,tf)
			return sol

	# integral TF**2 for del1!=del0	fast version
	def TF_del1_2(self,del0,del1,m):
		phi = lambda x: np.tanh(self.A * x)
		sd_0 = np.sqrt(del0) * self.xgrid
		sd_1 = np.sqrt(del0-del1) * self.xgrid
		TF = phi(np.add.outer(sd_1,sd_0) + m)
		sol = (self.dx**3) * np.einsum('i,j,ij,kj,k',self.normal_pdf, self.normal_pdf ,TF, TF,self.normal_pdf)
		return sol
		


class MFTCurves:
	'''This class provides the curves from the mft'''

	def __init__(self):
		self.GI = Integrals()
		# gamma
		self.A = 3
		self.GI.A = self.A
	
	def value_A(self,A):
		self.A = A
		self.GI.A = self.A

	
	def overlap(self,del0,m_init):
		''' This function gives de overlap
		for a given del0 and initial m'''
		m_min = 1e-2 # minimal m concidered zero
		max_iter = 5000 # maximum number of iterations
		m=m_init
		error=1.
		for i in range(max_iter):
			sol = self.GI.overlap(del0,m)
			error=np.abs(m-sol)
			m=sol
			if error<1e-7:
				return m
			if m<m_min:
				return m
		return m

	#critical overlap
	def critical_overlap(self):
		field = lambda x:self.GI.Doverlap(x)-1.
		try:
			del0_c = brentq(field,0,20.)
		except:
			del0_c = 0
		return del0_c

	# transition to chaos background state
	def chaos_background(self):
		# defining the model
		myfield = lambda x:self.GI.TF(x,0)/(self.GI.DTF(x,0))-x
		#chaos background
		del0 = brentq(myfield,0.,20.)
		alpha_c = 1./ self.GI.DTF(del0,0)
		return alpha_c

	#Transition to chaos fixed points
	def chaos_fixed_points(self,m_init=1.):
		# defining the model
		m  = lambda y: self.overlap(y,m_init) # the overlap dep del0
		field = lambda x:self.GI.TF(x,m(x))/self.GI.DTF(x,m(x))-x
		del0_c = brentq(field,0.0,20.,xtol=1e-4)
		m_c = m(del0_c)
		alpha_c = 1./self.GI.DTF(del0_c,m_c)
		return del0_c,m_c,alpha_c
	
	#computing from TT paper
	def capacity_SMFT(self):
		del0_c= self.critical_overlap()
		alpha_c = del0_c/self.GI.TF(del0_c,0)
		return del0_c,0,alpha_c
	
	#computing from TT paper
	def capacity_DMFT(self):
		del0= self.critical_overlap()
		m = 0
		alpha_c = ((del0 * self.A)**2)/(2 * (self.GI.Int_TF2(del0,m)-self.GI.Int_TF(del0,m)**2))
		return alpha_c
	
	def overlap_SMFT(self,alpha,q0=1.):
		''' Returns the overlap and del0 
		calculated from the static MFT'''
		m = lambda d0: self.overlap(d0,q0)
		myfield = lambda x:alpha*self.GI.TF(x,m(x))-x
		del0 = brentq(myfield,0,2000.)
		return del0,self.overlap(del0,q0)
	
	#overlap curve static dmft
	def overlap_DMFT(self,alpha,q0=1.,x0 = np.array([0.5,0.])):
		m = lambda d0: self.overlap(d0,q0)
		field = lambda x:np.array([self.f1(x[0],x[1],alpha,q0),self.f2(x[0],x[1],alpha,q0)])
		#sol = root(field,x0,method='hybr',options={'xtol': 1.e-04})
		sol = root(field,x0)
		del0 = sol.x[0]
		return sol.x,m(del0)


	def f2(self,del0,del1,alpha,q0):
		''' This function equalize potentials in the MFT'''
		s = 1e-2
		if del0 <0:
			return 1e6
		elif del1+s<del0:
			m = lambda d0: self.overlap(d0,q0)

			# potential at del1 = del0
			int0 = lambda d0:self.GI.Int_TF2(d0,m(d0))
			pot0 = lambda d0: -(d0**2)/2. +(1./self.A**2) * int0(d0) * alpha
		
			#potential at del1!=del0
			int1 = lambda d0,d1:self.GI.IntTF_del1(d0,d1,m(d0))
			pot1 = lambda d0,d1: -(d1**2)/2. + (1./self.A**2) * int1(d0,d1) * alpha
			
			f2 = lambda d0,d1:pot1(d0,d1)-pot0(d0)
			
			return f2(del0,del1)
		
		else:
			return 1e6
		
	def f1(self,del0,del1,alpha,q0):
		'''This function is the self-conistent equation for del1'''
		s = 1e-2
		if del0<=0:
			return 1e6
		elif del1+s<del0:
			# defining the model
			m = lambda d0: self.overlap(d0,q0)
			f1 = lambda d0,d1:alpha * self.GI.TF_del1(d0,d1,m(d0))-d1	
			
			return f1(del0,del1)
		else:
			return 1e6

	
	def autocov_curve(self,alpha, del0, del_init,  m):
		'''Solving the mechanical equation for the potential'''
		field1 = lambda x,y:y
		field2 = lambda x,y:x - alpha * self.GI.TF_del1(del0,x,m)	
		dt = 0.0005
		x_t = del_init
		y_t = 0
		dyn_del = []
		#37678 A = 5.5
		for l in range(37678):
			k1_x = dt * field1(x_t, y_t)
			k2_x = dt * field1(x_t + dt/2., y_t + k1_x/2.)
			k3_x = dt * field1(x_t + dt/2., y_t + k2_x/2.)
			k4_x = dt * field1(x_t + dt, y_t + k3_x)

			k1_y = dt * field2(x_t, y_t)
			k2_y = dt * field2(x_t + dt/2., y_t + k1_y/2.)
			k3_y = dt * field2(x_t + dt/2., y_t + k2_y/2.)
			k4_y = dt * field2(x_t + dt, y_t + k3_y)
			
			x_t= x_t + (1./6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
			y_t= y_t + (1./6) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)

			dyn_del.append(x_t)
			print('time (tau)=',round(dt * l,3),'|delt=', x_t,'|del0=', del0)
		return dyn_del

	
