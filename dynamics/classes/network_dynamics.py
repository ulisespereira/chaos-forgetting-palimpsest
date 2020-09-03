import numpy as np
from scipy import sparse 
from scipy.stats import bernoulli
from scipy import sparse 
from scipy import integrate
from scipy.optimize import brentq
import time




class TransferFunction:
    '''this class is for different transfer functions'''
    def __init__(self, modelparams):
        self.which_tf = modelparams['which_tf']
        self.rm = modelparams['r_max']
        self.b = modelparams['beta']
        self.h0 = modelparams['h0']	
        if self.which_tf=='tanh':
            self.q_tanh = modelparams['q_tanh']

    def TF(self,h):
        if self.which_tf=='sig':# sigmoidal TF
            phi = self.rm/(1.+np.exp(-self.b * (h-self.h0)))
        elif self.which_tf=='tanh':# tanh TF
            phi =  (self.rm/2.) * (2 * self.q_tanh-1. + np.tanh(self.b * (h - self.h0)))
        return phi

class LearningRule:
    ''' This class gives the learning rule'''
    def __init__(self, modelparams):
        self.myTF = TransferFunction(modelparams)
        self.myTF.b = 1e6
        #parameters function g and f
        self.xf = modelparams['xf']
        self.xg = modelparams['xg']
        self.betaf = modelparams['bf']
        self.betag = modelparams['bg']
        self.qf = modelparams['qf']
        self.Amp = modelparams['amp']
        self.scale = 2. # set it to two for TT model
        # here it is very important do it in this order
        self.qg = self.Qg()
        self.intg2 = self.Eg2()
        self.intf2 = self.Ef2()
        self.gamma = self.intf2 * self.intg2 * self.Amp**2
    
    def std_normal(self,x):
        sigma=1. # standard normal random variable passed thorugh transfer functiion
        mu=0
        pdf=(1./np.sqrt(2 * np.pi * sigma**2))*np.exp(-(1./2.)*((x-mu)/sigma)**2)
        return pdf
    
    # separable functions learning process leanring rule f and g
    def f(self, x):
        return  self.scale * 0.5 * (2*self.qf-1.+np.tanh(self.betaf*(x-self.xf)))
    
    def g(self, x):
        return  self.scale * 0.5 * (2*self.qg-1.+np.tanh(self.betag*(x-self.xg)))
    
    def Qg(self):# mean of f**2
        return brentq(self.Eg,0.,1.)
    
    def Eg(self,q):# mean of g
        self.qg = q
        fun=lambda x:self.std_normal(x)*self.g(self.myTF.TF(x))
        var,err=integrate.quad(fun,-10.,10.)
        return var
    
    def Ef2(self):# mean of f**2
        fun=lambda x:self.std_normal(x)*self.f(self.myTF.TF(x))*self.f(self.myTF.TF(x))
        var,err=integrate.quad(fun,-10.,10.)
        return var
    
    def Eg2(self):# mean of g**2
        fun=lambda x:self.std_normal(x)*self.g(self.myTF.TF(x))*self.g(self.myTF.TF(x))
        var,err=integrate.quad(fun,-10,10.)
        return var

class ConnectivityMatrix:
    '''This class creates the connectivity matrix'''
    def __init__(self,modelparams):
        self.seed = modelparams['seed']
        #tranfer function and learning rule
        self.myLR = LearningRule(modelparams)
        # parameters for the dynamics
        self.N = modelparams['N']
        self.sparsity()
        self.tau  = modelparams['tau_palimpsest']# time scale forgetting
        self.n_tau  = modelparams['n_tau_palimpsest']# time scale forgetting
        self.number_stored_patterns()
        #order is important
        self.forgetting_kernel()
        self._make_indexes_connectivity()
        self.dN = 300000 #sinze chunks connectivity
        self.n = int(self.N2bar/self.dN) #truncate number of chunks

    def make_patterns(self):
        '''make patterns and fixed the random seed'''
        np.random.seed(self.seed)
        patterns_fr = np.random.normal(0.,1., size=(self.p,self.N))
        return patterns_fr
    
    def sparsity(self):
        '''scalling sparsity network's connectivity'''
        self.c = 1./np.sqrt(self.N)
    
    def number_stored_patterns(self):
        '''p->inifty, in practice p = n * tau * K, with n big enough'''
        self.K = self.N * self.c
        self.p = int(round(self.K * self.tau * self.n_tau))
    
    def forgetting_kernel(self):
        ''' forgetting kernel'''
        self.forget = np.exp(- np.arange(0, self.p, 1)/(self.tau * self.K))

    def _make_pre_post_patterns(self):
        ''' make the pre and post synaptic patterns
        using a generalized Hebbian learning rule
        and the forgetting kernel'''    
        patterns_fr = self.make_patterns()
        patterns_pre = self.myLR.g(patterns_fr)
        patterns_pre = np.einsum('ij,i->ij', patterns_pre, self.forget)
        patterns_post = self.myLR.f(patterns_fr)
        print('Patterns created. N patterns:', self.p)
        return patterns_pre, patterns_post

    def _make_indexes_connectivity(self):
        #number of entries different than zero
        self.N2bar = np.random.binomial(self.N * self.N, self.c)
        self.row_ind = np.random.randint(0, high = self.N, size = self.N2bar)
        self.column_ind = np.random.randint(0, high = self.N, size = self.N2bar)
        print('Structural connectivity created')
    
    def _chunk_connectivity(self, l, patterns_pre, patterns_post):
        ''' make chunk connectivity'''
        con_chunk = np.einsum('ij,ij->j', patterns_post[:, self.row_ind[l * self.dN:(l+1) * self.dN]], patterns_pre[:, self.column_ind[l * self.dN:(l+1) * self.dN]])
        return con_chunk
        

    def connectivity_generalized_hebbian(self):
        patterns_pre, patterns_post = self._make_pre_post_patterns()
        connectivity=np.array([])
        for l in range(self.n):
            con_chunk = self._chunk_connectivity(l, patterns_pre, patterns_post)
            # smart way to write down the outer product learning
            connectivity = np.concatenate((connectivity, con_chunk), axis=0)
            
            print('Synaptic weights created:',np.round(100.*(l)/float(self.n),3),'%')
        con_chunk = np.einsum('ij,ij->j', patterns_post[:, self.row_ind[self.n * self.dN:self.N2bar]], patterns_pre[:, self.column_ind[self.n * self.dN:self.N2bar]])
        print('Synaptic weights created:',100.,'%')
        connectivity = np.concatenate((connectivity, con_chunk),axis=0)	    
        connectivity = (self.myLR.Amp/self.K) * connectivity
        print('Synaptic weights created')
        connectivity = sparse.csr_matrix((connectivity,(self.row_ind, self.column_ind)), shape=(self.N,self.N))
        print('connectivity created')
        self.con_matrix =  connectivity
	
		

class NetworkDynamics:
    
    '''This class creates the connectivity matrix'''
    
    def __init__(self, parameters_values, connectivity):
        # dynamics
        #tranfer function and learning rule
        self.myLR = connectivity.myLR
        self.myTF = TransferFunction(parameters_values)
        self.N = parameters_values['N']
        self.dt = parameters_values['dt']# dt integration
        self.T = parameters_values['T']
        self.tau = parameters_values['tau'] # 20ms 
        #input current
        self.Input = 0.
        self.con_matrix = connectivity.con_matrix
        self.patterns_fr = connectivity.make_patterns()
        self.neu_indexes = parameters_values['indexes_neurons']
    
    def fieldDynamics(self, u, t):
        return (1./self.tau)*(-u+self.Input+self.con_matrix.dot(self.myTF.TF(u)))#-1.*np.mean(u)))
    
    def _overlaps(self, rn, index): 
        ''' computing the overlaps'''
        overlap = (1./self.N) * np.einsum('ij,j->i',self.myLR.g(self.patterns_fr[index[0]:index[1], :]), rn)
        return overlap

    def dynamics(self, u_init, index):
        un = u_init #initial condition
        p, N = self.patterns_fr.shape
        mysol = [] #neurons dynammics
        q_ord_p = [] # overlap
        del0_ord_p = [] # del0
        rn = self.myTF.TF(un)
        mysol.append(un[self.neu_indexes])
        overlap = self._overlaps(rn, index)
        q_ord_p.append(overlap)
        del0 = np.mean((un-np.mean(un)) * (un - np.mean(un)))
        del0_ord_p.append(del0)
        t=0
        while t<=self.T:
            un = un + self.dt * self.fieldDynamics(un,t)
            t = t + self.dt
            mysol.append(un[self.neu_indexes])
            rn =  self.myTF.TF(un)
            # calculating order parameters
            overlap = self._overlaps(rn, index)
            q_ord_p.append(overlap)
            del0 = np.mean((un-np.mean(un)) * (un - np.mean(un)))
            del0_ord_p.append(del0)
            if t%500==0:
                print('time=', t, ' of T=', self.T, '|del0=', round(del0,2), '|overlap=', round(overlap[0], 2))
        sol = dict(
                overlaps = np.array(q_ord_p),
                del0s = np.array(del0_ord_p),
                dyn = np.array(mysol)
                )

        return sol

