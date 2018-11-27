'''
Created on 19 Sep 2017

@author: richard
'''
import numpy as np
import scipy.sparse as sp
import quant_mech.utils as utils
from quant_mech.OBOscillator import OBOscillator
from quant_mech.UBOscillator import UBOscillator
from quant_mech.hierarchy_solver import HierarchySolver

class PhotocellModel(object):
    
    def __init__(self, delta_E,
                       J,
                       drude_reorg_energy,
                       drude_cutoff,
                       mode_freq,
                       mode_S,
                       mode_damping,
                       CT_scaling,
                       temperature=300.,
                       N=6,
                       K=0,
                       v=1.,
                       no_mode=False):
        '''
            Parameters...
            delta_E: unshifted electronic energy splitting in site basis
            J: electronic coupling 
            drude_reorg_energy: reorg energy due to Drude bath
            drude_cutoff: relaxation parameter of the Drude bath
            mode_freq: frequency of the mode
            mode_S: Huang-Rhys factor of the mode
            mode_damping: damping constant for the mode
            CT_scaling: the scaling parameter for the CT state reorg energy
            temperature
            N: number of tiers to use with HEOM
            K: number of Matsubara terms to use in correlation function expansion with HEOM
            v: the scaling parameter for the excited state reorg energy
            no_mode: whether to include the mode in the spectral density or not
        '''
    
        self.el_dim = 5 # the dimension of the electronic system
        
        self.delta_E = delta_E # 100.
        self.J = J # 5.
        
        #evals,evecs = np.linalg.eig(H)
        
        self.T = temperature # 300. # temperature in Kelvin
        self.beta = 1. / (utils.KELVIN_TO_WAVENUMS * self.T) # the inverse temperature in wavenums
        
        self.drude_reorg_energy = drude_reorg_energy # 35.
        self.drude_cutoff = drude_cutoff # 40.
        
        self.no_mode = no_mode
        self.mode_freq = mode_freq # 342.
        self.mode_S = mode_S # 0.0438
        self.mode_damping = mode_damping # 10.
        self.mode_reorg_energy = self.mode_freq * self.mode_S
        
        self.total_reorg_energy = (self.drude_reorg_energy + self.mode_reorg_energy) if not self.no_mode else self.drude_reorg_energy
        
        self.v = v # scaling of site 1 reorg energy (in case we want to vary it too)
        self.c = CT_scaling # scaling of CT state reorg energy
        
        self.N = N # truncation level
        self.K = K # num Matsubara terms
        
        '''
        Where did I take the following parameters from?
        '''        
        self.n_ex = 6000 # number of photons in the 'hot' bath
        self.gamma = 1.e-2 # bare transition rate between ground and excited states 
        self.gamma_ex = self.gamma * self.n_ex # rate from ground to excited state
        self.gamma_deex = self.gamma * (self.n_ex + 1.) # rate from excited to ground state
        
        self.secondary_CT_energy_gap = 300.
        self.bare_secondary_CT = 1. #0.0025 # rate from CT1 to CT2
        self.n_c = utils.planck_distribution(self.secondary_CT_energy_gap, self.T)
        self.forward_secondary_CT = (self.n_c + 1) * self.bare_secondary_CT
        self.backward_secondary_CT = self.n_c * self.bare_secondary_CT
        
        self.Gamma_L = 1. # 0.025 # the rate from left lead to populate ground state
        self.Gamma_R = 1. # rate from CT2 state to right lead
        
        # set up environment depending on no_mode
        if not self.no_mode:
            env = [(),
                   (OBOscillator(self.v*self.drude_reorg_energy, self.drude_cutoff, beta=self.beta, K=self.K),
                    UBOscillator(self.mode_freq, self.v*self.mode_S, self.mode_damping, beta=self.beta, K=self.K)),
                   (OBOscillator(self.c*self.drude_reorg_energy, self.drude_cutoff, beta=self.beta, K=self.K),
                    UBOscillator(self.mode_freq, self.c*self.mode_S, self.mode_damping, beta=self.beta, K=self.K)),
                   (),
                   ()]
        else:
            env = [(),
                   (OBOscillator(self.v*self.drude_reorg_energy, self.drude_cutoff, beta=self.beta, K=self.K),),
                   (OBOscillator(self.c*self.drude_reorg_energy, self.drude_cutoff, beta=self.beta, K=self.K),),
                   (),
                   ()]
        
        # I think this dummy solver is set up so we can get info about the size of hierarchy 
        # etc using methods from HierarchySolver before actually creating the hierarchy matrix
        self.dummy_solver = HierarchySolver(self.el_hamiltonian(), env, self.beta, N=self.N, num_matsubara_freqs=self.K)
        
    def el_hamiltonian(self):
        '''
        The electronic Hamiltonian. We can ignore the energies of ground, CT2 and
        empty states as there is no coherent evolution between them. We take into
        account the relative energies through the incoherent rates instead. 
        '''
        H = np.zeros((self.el_dim, self.el_dim))
        H[1:3,1:3] = np.array([[self.delta_E/2., self.J],
                               [self.J, -self.delta_E/2.]]) + np.diag([self.v*self.total_reorg_energy, self.c*self.total_reorg_energy])
        return H
        
    def excitation_op(self):
        '''
        Operator to take the system from ground state to site 1 (excited state)
        transformed to the electronic eigenstate basis.
        '''
        op = np.zeros((self.el_dim, self.el_dim))
        op[1,0] = 1. # eigenstate basis
        
        evals,evecs = np.linalg.eig(self.el_hamiltonian()[1:3,1:3])
        transform = np.eye(self.el_dim, dtype='complex128')
        transform[1:3,1:3] = evecs
        
        return np.dot(transform.T, np.dot(op, transform))
    
    def deexcitation_op(self):
        '''
        Operator to take the system from site 1 (excited state) to ground state
        transformed to the electronic eigenstate basis.
        '''
        op = np.zeros((self.el_dim, self.el_dim))
        op[0,1] = 1. # eigenstate basis
        
        evals,evecs = np.linalg.eig(self.el_hamiltonian()[1:3,1:3])
        transform = np.eye(self.el_dim, dtype='complex128')
        transform[1:3,1:3] = evecs
        
        return np.dot(transform.T, np.dot(op, transform))
    
    def forward_secondary_CT_op(self):
        op = np.zeros((self.el_dim, self.el_dim))
        op[3,2] = 1.
        return op
    
    def backward_secondary_CT_op(self):
        op = np.zeros((self.el_dim, self.el_dim))
        op[2,3] = 1.
        return op
    
    def drain_lead_op(self):
        op = np.zeros((self.el_dim, self.el_dim))
        op[4,3] = 1.
        return op
    
    def source_lead_op(self):
        op = np.zeros((self.el_dim, self.el_dim))
        op[0,4] = 1.
        return op
    
    def construct_heom_matrix(self):
        
        if not self.no_mode:
            env = [(),
                   (OBOscillator(self.v*self.drude_reorg_energy, self.drude_cutoff, beta=self.beta, K=self.K),
                    UBOscillator(self.mode_freq, self.v*self.mode_S, self.mode_damping, beta=self.beta, K=self.K)),
                   (OBOscillator(self.c*self.drude_reorg_energy, self.drude_cutoff, beta=self.beta, K=self.K),
                    UBOscillator(self.mode_freq, self.c*self.mode_S, self.mode_damping, beta=self.beta, K=self.K)),
                   (),
                   ()]
        else:
            env = [(),
                   (OBOscillator(self.v*self.drude_reorg_energy, self.drude_cutoff, beta=self.beta, K=self.K),),
                   (OBOscillator(self.c*self.drude_reorg_energy, self.drude_cutoff, beta=self.beta, K=self.K),),
                   (),
                   ()]
        
        
        jump_operators = [self.excitation_op(), self.deexcitation_op(), \
                          self.forward_secondary_CT_op(), self.backward_secondary_CT_op(), \
                          self.drain_lead_op(), self.source_lead_op()]
        
        jump_rates = np.array([self.gamma_ex, self.gamma_deex, self.forward_secondary_CT, self.backward_secondary_CT, \
                      self.Gamma_R, self.Gamma_L])
        
        self.solver = HierarchySolver(self.el_hamiltonian(), environment=env, beta=self.beta, \
                                 jump_operators=jump_operators, jump_rates=jump_rates, \
                                 N=self.N, num_matsubara_freqs=self.K, temperature_correction=True)
        
        return self.solver.construct_hierarchy_matrix_super_fast()
    
    def jump_matrix(self):
        dim = self.dummy_solver.M_dimension()
        jop = sp.csr_matrix((dim,dim))
        jop[:self.el_dim**2,:self.el_dim**2] = self.Gamma_R * np.kron(self.drain_lead_op(), self.drain_lead_op())
        return jop
    
    def dv_pops(self):
        dv_pops = np.zeros(self.dummy_solver.system_dimension**2 * self.dummy_solver.number_density_matrices())
        dv_pops[:self.dummy_solver.system_dimension**2] = np.eye(self.dummy_solver.system_dimension).flatten()
        return dv_pops
    
    def update_Gamma_R(self, Gamma_R):
        self.Gamma_R = Gamma_R
        self.solver.jump_rates[-2] = Gamma_R
    
    
    
    
        
        
        
            