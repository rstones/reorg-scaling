'''
Created on 22 Sep 2017

@author: richard
'''
import numpy as np
from photocell_model import PhotocellModel
from counting_statistics.sparse.fcs_solver import FCSSolver

def site_basis_energy_and_coupling(Delta_E, theta):
    epsilon = (Delta_E*theta) / np.sqrt(1. + theta**2)
    J = epsilon / (2. * theta)
    return epsilon, J

delocalization_values = np.logspace(-1, 1, 20)
scaling_values = np.linspace(0.5, 4, 20)

current = np.zeros((delocalization_values.size, scaling_values.size))
F2 = np.zeros((delocalization_values.size, scaling_values.size))

# delta_E = 340.
# J = 17.
drude_reorg_energy = 35.
drude_cutoff = 40.
mode_freq = 342.
mode_S = 0.0438
mode_damping = 10.

eigenbasis_energy_splitting = 340.

Gamma_R = 86143.6936712

T = 300.

for i,theta in enumerate(delocalization_values):
    print(theta)
    delta_E, J = site_basis_energy_and_coupling(eigenbasis_energy_splitting, theta)
    
    for j,c in enumerate(scaling_values):
        
        model = PhotocellModel(delta_E, J, drude_reorg_energy, drude_cutoff, mode_freq, mode_S, mode_damping, \
                               c, temperature=T, N=5, K=0, v=1)
        model.Gamma_R = Gamma_R
        H = model.construct_heom_matrix()
        jump_op = model.jump_matrix()
        pops = model.dv_pops()
        solver = FCSSolver(H, jump_op, pops)
        current[i,j] = solver.mean()
        F2[i,j] = solver.second_order_fano_factor()

np.savez('../../../data/heom_photocell_current_F2_delocalization_scaling_resonant_mode_log.npz',\
         delocalization_values=delocalization_values, scaling_values=scaling_values, \
         current=current, F2=F2)


