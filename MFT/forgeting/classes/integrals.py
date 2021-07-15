import numpy as np
from scipy.stats import multivariate_normal as mvnormal
import scipy.integrate as integrate

class Integrals:
	'''This is a class where appear all the TT integrals'''

	def __init__(self,A,tau,a):
		self.A = A	
		self.tau = tau
		self.a = a
		self.gam = self.Gamma()
		
		# grid normal integration
		self.dx=0.01 #
		x_min=-10.
		x_max=10.
		self.xgrid = np.arange(x_min,x_max,self.dx) #grid on x
		self.normal_pdf =self.std_normal(self.xgrid) #standard normal pdf
	
	def kernel2(self,t):
		return np.exp(-(2 * t)/self.tau) * (t + 1)**(2 * self.a)
	
	def kernel(self,t):
		return np.exp(-t/self.tau) * (t + 1)**self.a

	def std_normal(self,x):
		sigma=1. # standard normal random variable passed thorugh transfer functiion
		mu=0
		pdf=(1./np.sqrt(2 * np.pi * sigma**2))*np.exp(-(1./2.)*((x-mu)/sigma)**2)
		return pdf
	
	def Gamma(self):
		#dx = self.tau/100.
		#x_max = 200 * self.tau 
		#xgrid = np.arange(0,x_max,dx) #grid on x
		#sol = dx * np.sum(self.kernel(xgrid))
		sol,err  = integrate.quad(self.kernel2,0,np.inf)
		return sol





	# Doverlap TT	
	def Doverlap(self,del0,t):	
		Dphi = lambda x: self.A * self.kernel(t) * (1/np.cosh(self.A * x))**2
		sdfield = np.sqrt(del0) * self.xgrid
		DTF = Dphi(sdfield)
		sol = self.dx * np.einsum('i,i',DTF,self.normal_pdf)
		return sol

	# overlap TT	
	def overlap(self,del0,m,t):	
		phi = lambda x: np.tanh(self.A * x)
		sdfield = np.sqrt(del0) * self.xgrid
		TF = phi(m * self.kernel(t) + sdfield)
		sol = self.dx * np.einsum('i,i',TF,self.normal_pdf)
		return sol

	# TF * TF TT	
	def TF(self,del0,m,t):	
		phi = lambda x: np.tanh(self.A * x)
		sdfield = np.sqrt(del0) * self.xgrid
		TF2 = phi(sdfield + m * self.kernel(t))**2
		sol = self.dx * np.einsum('i,i',TF2,self.normal_pdf)
		return sol
	
	# DTF * DTF  TT	
	def DTF(self,del0,m,t):	
		Dphi = lambda x: self.A * (1/np.cosh(self.A * x))**2
		sdfield = np.sqrt(del0) * self.xgrid
		DTF2 = Dphi(sdfield + m * self.kernel(t))**2
		sol = self.dx * np.einsum('i,i',DTF2,self.normal_pdf)
		return sol

	# TF * TF TT	
	def Int_TF2(self,del0,m,t):	
		int_phi = lambda x: np.log(np.cosh(self.A * x))
		sdfield = np.sqrt(del0) * self.xgrid
		INT_TF2 = int_phi(sdfield + m * self.kernel(t))**2
		sol = self.dx * np.einsum('i,i',INT_TF2,self.normal_pdf)
		return sol
	# TF * TF TT	
	def Int_TF(self,del0,m,t):	
		int_phi = lambda x: np.log(np.cosh(self.A * x))
		sdfield = np.sqrt(del0) * self.xgrid
		INT_TF = int_phi(sdfield+m * self.kernel(t))
		sol = self.dx * np.einsum('i,i',INT_TF,self.normal_pdf)
		return sol

	# indegral d0 and d1 chaos
	def TF_d1(self,del0,del1,m,t):	
		if del0-del1<=1e-20:
			return self.TF(del0,m,t)
		else:
			phi = lambda x: np.tanh(self.A * x)
			
			
			d0_d1 = np.sqrt(del0-del1) * self.xgrid
			d1 = np.sqrt(del1) *  self.xgrid
			sdf = np.add.outer(d0_d1,d1)
			
			
			TF = phi(sdf + m * self.kernel(t))
			
			s_int = self.dx * np.einsum('ij,i->j',TF,self.normal_pdf)
			int_p = self.dx * np.einsum('i,i,i',s_int,s_int,self.normal_pdf)


			return int_p 

	# indegral d0 and d1 chaos
	def Int_TF2_d1(self,del0,del1,m,t):	
		if del0-del1<=1e-20:
			return self.Int_TF2(del0,m,t)
		else:
			int_phi = lambda x: np.log(np.cosh(self.A * x))
			
			
			d0_d1 = np.sqrt(del0-del1) * self.xgrid
			d1 = np.sqrt(del1) *  self.xgrid
			sdf = np.add.outer(d0_d1,d1)
			
			
			INT_TF = int_phi(sdf + m * self.kernel(t))
			
			s_int = self.dx * np.einsum('ij,i->j',INT_TF,self.normal_pdf)
			int_p = self.dx * np.einsum('i,i,i',s_int,s_int,self.normal_pdf)


			return int_p 
