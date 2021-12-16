import numpy as np
import scipy.integrate as integrate
from scipy.optimize import brentq,root,fmin_tnc
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

    # overlap two-states	
    def overlap_mixed(self, del0 ,m1, t1, m2, t2):	
        phi = lambda x: np.tanh(self.A * x)
        sdfield = np.sqrt(del0) * self.xgrid
        TF1 = phi(m1 * self.kernel(t1) + m2 * self.kernel(t2) +  sdfield)
        TF2 = phi(m1 * self.kernel(t1) - m2 * self.kernel(t2) +  sdfield)
        TF = (TF1 + TF2)/2.
        sol = self.dx * np.einsum('i,i',TF,self.normal_pdf)
        return sol
    
    # TF * TF two-states	
    def TF_mixed(self,del0, m1, t1, m2, t2):	
        phi = lambda x: np.tanh(self.A * x)
        sdfield = np.sqrt(del0) * self.xgrid
        TF2_1 = phi(sdfield + m1 * self.kernel(t1) + m2 * self.kernel(t2))**2
        TF2_2 = phi(sdfield + m1 * self.kernel(t1) - m2 * self.kernel(t2))**2
        TF2 = (TF2_1 + TF2_2)/2.
        sol = self.dx * np.einsum('i,i', TF2, self.normal_pdf)
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
    def Int_TF2_mixed(self,del0, m1, t1, m2, t2):	
        int_phi = lambda x: np.log(np.cosh(self.A * x))
        sdfield = np.sqrt(del0) * self.xgrid
        INT_TF2_1 = int_phi(sdfield + m1 * self.kernel(t1) + m2 * self.kernel(t2))**2
        INT_TF2_2 = int_phi(sdfield + m1 * self.kernel(t1) - m2 * self.kernel(t2))**2
        sol_1 = self.dx * np.einsum('i,i', INT_TF2_1, self.normal_pdf)
        sol_2 = self.dx * np.einsum('i,i', INT_TF2_2, self.normal_pdf)
        sol =  (sol_1 + sol_2)/2.
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
    def TF_d1_mixed(self,del0, del1, m1, t1, m2, t2):	
        if del0-del1<=1e-20:
            return self.TF_mixed(del0, m1, t1, m2, t2) 
        else:
            phi = lambda x: np.tanh(self.A * x)
            
            
            d0_d1 = np.sqrt(del0-del1) * self.xgrid
            d1 = np.sqrt(del1) *  self.xgrid
            sdf = np.add.outer(d0_d1,d1)
            
            
            TF_1 = phi(sdf + m1 * self.kernel(t1) + m2 * self.kernel(t2))
            TF_2 = phi(sdf + m1 * self.kernel(t1) - m2 * self.kernel(t2))

            s_int_1 = self.dx * np.einsum('ij,i->j',TF_1,self.normal_pdf)
            s_int_2 = self.dx * np.einsum('ij,i->j',TF_2,self.normal_pdf)
            
            int_p_1 = self.dx * np.einsum('i,i,i',s_int_1, s_int_1, self.normal_pdf)
            int_p_2 = self.dx * np.einsum('i,i,i',s_int_2, s_int_2,self.normal_pdf)

            int_p = (int_p_1 + int_p_2)/2.

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

    # indegral d0 and d1 chaos
    def Int_TF2_d1_mixed(self, del0, del1, m1, t1, m2, t2):	
        if del0-del1<=1e-20:
            return self.Int_TF2_mixed(del0, m1, t1, m2, t2) 
        else:
            int_phi = lambda x: np.log(np.cosh(self.A * x))
            
            
            d0_d1 = np.sqrt(del0-del1) * self.xgrid
            d1 = np.sqrt(del1) *  self.xgrid
            sdf = np.add.outer(d0_d1,d1)
            
            
            INT_TF1 = int_phi(sdf + m1 * self.kernel(t1) + m2 * self.kernel(t2))
            INT_TF2 = int_phi(sdf + m1 * self.kernel(t1) - m2 * self.kernel(t2))
            #INT_TF3 = int_phi(sdf - m1 * self.kernel(t1) + m2 * self.kernel(t2))
            #INT_TF4 = int_phi(sdf - m1 * self.kernel(t1) + m2 * self.kernel(t2))

            s_int_1 = self.dx * np.einsum('ij,i->j',INT_TF1, self.normal_pdf)
            int_p_1 = self.dx * np.einsum('i,i,i',s_int_1, s_int_1, self.normal_pdf)

            s_int_2 = self.dx * np.einsum('ij,i->j',INT_TF2, self.normal_pdf)
            int_p_2 = self.dx * np.einsum('i,i,i',s_int_2, s_int_2, self.normal_pdf)
            
            int_p =  (int_p_1 + int_p_2)/2.
            return int_p 

class MFTCurves:
    '''This class provides the curves from the mft'''
    def __init__(self,A,tau,a):
        self.GI = Integrals(A,tau,a)
        # gamma
        self.A = A
        self.tau = tau
        self.a = a #option for a non-monotonic kernel
        self.gam = self.GI.gam
        #critical overlap
    
    def critical_overlap(self,t):
        '''Computing the overlap'''
        field = lambda x:self.GI.Doverlap(x,t)-1.
        try:
            del0_c = brentq(field,0,1.)
        except:
            del0_c = 0
            #print del0_c
        return del0_c

    def chaos_fixed_points(self):
        '''Transition to chaos fixed points'''
        # line search approach
        # defining the model
	#d0_c,t_c = self.capacity_SMFT(tau)
        time = np.arange(0.4,0,-0.001)
        ft = lambda d0,m,t:self.gam * self.GI.DTF(d0,m,t) - 1
        d0,m =  self.overlap_SMFT(time[0])
        error = ft(d0,m,time[0])
        for t in time:
            d0,m =  self.overlap_SMFT(t)
            if ft(d0,m,t) * error <= 0:
                return t
            error = ft(d0,m,t)	
        return t
	
    def capacity_SMFT(self,x0 = [1.,0]):
        '''capacity from the SMFT'''
        # x[0] = del0
        #x[1] = t
        field = lambda x:np.array([x[0] - self.gam * self.GI.TF(x[0],0,x[1]),self.GI.Doverlap(x[0],x[1])-1.])
        #t_c = brentq(field,0,0.5)
        sol = root(field, x0, method='hybr')
        return sol.x[0],sol.x[1]
    
    def capacity_DMFT(self, x0 = [1., 0]):
        '''capacity from the DMFT'''
        # x[0] = del0
        #x[1] = t
        def field(x):
            field1 = -x[0]**2/2. + (self.gam/(self.A**2)) * (self.GI.Int_TF2(x[0],0,x[1])-self.GI.Int_TF(x[0],0,x[1])**2)
            field2 = self.GI.Doverlap(x[0], x[1]) - 1.
            return np.array([field1, field2]) 
        sol = root(field, x0, method='hybr')
        return sol.x[0], sol.x[1]
	
    #overlap curve static Dft
    def overlap_DMFT_mixed(self, t, t0 = 0):

        # defining the model 
        dt1 = 0.1
        dt2 = 0.1
        dt3 = 0.1
        
        m1 = 1.
        m2 = 0.1
        d0 = 0.0
        d1 = 0.0
        error=1.
        
        max_iter = 10000000
        
        fm1 = lambda d0, d1, m1, t1, m2, t2:self.GI.overlap_mixed(d0, m1, t1, m2, t2)
        fm2 = lambda d0, d1, m1, t1, m2, t2:self.GI.overlap_mixed(d0, m1, t1, m2, t2) 
        def fd0(d0, d1, m1, t1, m2, t2):
            return np.sqrt( ((2 * self.gam)/(self.A**2)) * (self.GI.Int_TF2_mixed(d0, m1, t1, m2, t2) - self.GI.Int_TF2_d1_mixed(d0, d1, m1, t1, m2, t2)) + d1**2)
        fd1 = lambda d0, d1, m1, t1, m2, t2:self.gam * self.GI.TF_d1_mixed(d0, d1, m1, t1, m2, t2)

        for i in range(max_iter):
            f_m1 = fm1(d0, d1, m1, t, m2, t0)
            f_m2 = fm2(d0, d1, m2, t0, m1, t)
            f_d0 = fd0(d0, d1, m1, t, m2, t0)
            f_d1 = fd1(d0, d1, m1, t, m2, t0)
            error = max([abs(m1 - f_m1), abs(m2 - f_m2), abs(d0 - f_d0), abs(d1 - f_d1)])
            #print 'd0=',d0,'d1=',d1,'m=',m,'error',error
            if error<1e-4:
                m1 = m1 + dt1 * (f_m1 - m1)
                m2 = m2 + dt1 * (f_m2 - m2)
                d0 = d0 + dt2 * (f_d0 - d0)
                d1 = d1 + dt3 * (f_d1 - d1)
                return d0, d1, m1, m2
            #print(i, error, m1, m2)
            if m1<1e-2: #after capacity
                #self.overlap_DMFT_mixed(self, t0):
                return d0, d1, 0, m2
            m1 = m1 + dt1 * (f_m1 - m1)
            m2 = m2 + dt2 * (f_m2 - m2)
            d0 = d0 + dt2 * (f_d0 - d0)
            d1 = d1 + dt3 * (f_d1 - d1)
        return d0, d1, m1, m2

    #overlap curve static Dft
    def overlap_DMFT(self, t):

        # defining the model 
        t1 = 0.1
        t2 = 0.1
        t3 = 0.1
        
        m = 1.
        d0 = 0.0
        d1 = 0.0
        error=1.
        
        max_iter = 10000000
        
        fm = lambda d0, d1, m, t:self.GI.overlap(d0, m, t)
        fd0 = lambda d0, d1, m, t:np.sqrt( ((2 * self.gam)/(self.A**2)) * (self.GI.Int_TF2(d0, m, t)-self.GI.Int_TF2_d1(d0, d1, m, t)) + d1**2)
        fd1 = lambda d0, d1, m, t:self.gam * self.GI.TF_d1(d0, d1, m, t)

        for i in range(max_iter):
            f_m = fm(d0, d1, m, t)
            f_d0 =  fd0(d0, d1, m, t)
            f_d1 =  fd1(d0, d1, m, t)
            error = max([abs(m - f_m), abs(d0 - f_d0), abs(d1 - f_d1)])
            #print 'd0=',d0,'d1=',d1,'m=',m,'error',error
            if error<1e-4:
                m = m + t1 * (f_m - m)
                d0 = d0 + t2 * (f_d0 - d0)
                d1 = d1 + t3 * (f_d1 - d1)
                return d0, d1, m
            if m<1e-2: #after capacity
                return d0,d1,0
            m = m + t1 * (f_m - m)
            d0 = d0 + t2 * (f_d0 - d0)
            d1 = d1 + t3 * (f_d1 - d1)
        return d0, d1, m

    #overlap curve static mft
    def overlap_SMFT(self,t):
        # defining the model
        
        t1 = 0.1
        t2 = 0.1
        
        m = 1.
        d0 = 0.
        error = 1.
        max_iter = 10000000

        fm = lambda d0, m, t:self.GI.overlap(d0, m, t)
        fd0 = lambda d0, m, t:self.GI.TF(d0, m, t)

        for i in range(max_iter):
            f_m = fm(d0, m, t)
            f_d0 = self.gam *  fd0(d0, m, t)
            error = max([abs(m - f_m), abs(d0 - f_d0)])
            if error<1e-6:
                m = m + t1 * (f_m-m)
                d0 = d0 + t2 * (f_d0-d0)
                return d0, m
            if m<1e-2: #after capacity
                return d0, 0
            m = m + t1 * (f_m-m)
            d0 = d0 + t2 * (f_d0-d0)
        return d0, m
    
    #overlap curve static mft
    def overlap_SMFT_mixed(self,t, t0=0):
        # defining the model
        
        dt1 = 0.1
        dt2 = 0.1
        dt3 = 0.1
        
        m1 = 1.
        m2 = 0.1
        d0 = 0.
        error = 1.
        max_iter = 10000000

        fm1 = lambda d0, m1, t1, m2, t2:self.GI.overlap_mixed(d0, m1, t1, m2, t2)
        fm2 = lambda d0, m1, t1, m2, t2:self.GI.overlap_mixed(d0, m1, t1, m2, t2)
        fd0 = lambda d0, m1, t1, m2, t2:self.GI.TF_mixed(d0, m1, t1, m2, t2)
        
        for i in range(max_iter):
            f_m1 = fm1(d0, m1, t, m2, t0)
            f_m2 = fm2(d0, m2, t0, m1, t)
            f_d0 = self.gam *  fd0(d0, m1, t, m2, t0)
            error = max([abs(m1 - f_m1), abs(m2 - f_m2), abs(d0 - f_d0)])
            if error<1e-6:
                m1 = m1 + dt1 * (f_m1 - m1)
                m2 = m2 + dt2 * (f_m2 - m2)
                d0 = d0 + dt2 * (f_d0 - d0)
                return d0, m1, m2
            if m1<1e-2: #after capacity
                return d0, 0, m2
            m1 = m1 + dt1 * (f_m1 - m1)
            m2 = m2 + dt2 * (f_m2 - m2)
            d0 = d0 + dt2 * (f_d0 - d0)
        return d0, m1, m2
