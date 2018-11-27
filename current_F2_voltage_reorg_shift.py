'''
Created on 3 Oct 2017

@author: richard
'''
import numpy as np
import scipy.constants as constants
import quant_mech.utils as utils
import quant_mech.time_utils as tutils
from photocell_model import PhotocellModel
from counting_statistics.sparse.fcs_solver import FCSSolver
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence


def calculate_voltage(alpha_energy, beta_energy, temperature, alpha_pop, beta_pop):
    return ((alpha_energy-beta_energy)*utils.EV_TO_JOULES + (constants.k * temperature)*np.log(alpha_pop/beta_pop)) / constants.e

def extract_system_density_matrix(hierarchy_vector, sys_dim):
    sys_dm = hierarchy_vector[:sys_dim**2]
    sys_dm.shape = sys_dim, sys_dim
    return sys_dm

Gamma_R_range = np.logspace(-20., 5., num=20, base=np.exp(1.))*utils.EV_TO_WAVENUMS

'''
OFF-RESONANT MODE
delocalized: delta_E = 20, J = 100
intermediate: delta_E = 140, J = 70
localized: delta_E = 200, J = 10

RESONANT MODE
delocalized: delta_E = 34, J = 170
intermediate: delta_E = 240, J = 120
localized: delta_E = 340, J = 17
'''

# params = [(200., 5., 6.9),
#           (200., 5., 1.),
#           (340., 8.5, 10.),
#           (340., 8.5, 1.),
#           (140., 70., 3.),
#           (140., 70., 1.),
#           (240., 120., 5.),
#           (240., 120., 1.)]

# params = [(140., 70., 3.16, False),
#           (140., 70., 1., False),
#           (240., 120., 4.69, False),
#           (240., 120., 1., False),
#           (200., 5., 6.72, True),
#           (200., 5., 1., True),
#           (340., 8.5, 10.72, True),
#           (340., 8.5, 1., True)]

# no mode params asymmetric
# params = [(140., 70., 3.16, False),
#           (140., 70., 3.16, True),
#           (240., 120., 4.69, False),
#           (240., 120., 4.69, True),
#           #(200., 5., 6.72, False),
#           #(200., 5., 6.72, True),
#           (340., 8.5, 10.72, False),
#           (340., 8.5, 10.72, True)]
# 
# # no mode params symmetric
# params = [(140., 70., 1, False),
#           (140., 70., 1, True),
#           (240., 120., 1, False),
#           (240., 120., 1, True),
#           #(200., 5., 1, False),
#           #(200., 5., 1, True),
#           (340., 8.5, 1, False),
#           (340., 8.5, 1, True)]

params = [(240., 120., 4.69, False),
          (240., 120., 4.69, True),
          (340., 8.5, 10.72, False),
          (340., 8.5, 10.72, True),
          (240., 120., 1, False),
          (240., 120., 1, True),
          (340., 8.5, 1, False),
          (340., 8.5, 1, True)]


drude_reorg_energy = 35.
drude_cutoff = 40.
mode_freq = 342.
mode_S = 0.0438
mode_damping = 10.
mode_reorg_energy = mode_freq * mode_S
total_reorg_energy = drude_reorg_energy + mode_reorg_energy

T = 300.

for param_set in params:
    print(param_set)
    print(tutils.getTime())
    
    delta_E = param_set[0]
    J = param_set[1]
    c = param_set[2]
    no_mode = param_set[3]
    c_label = '_symmetric_' if c==1 else '_asymmetric_'
    
    num_tiers = 5 
    model = PhotocellModel(delta_E, J, drude_reorg_energy, drude_cutoff, mode_freq, mode_S, mode_damping, \
                           c, temperature=T, N=num_tiers, K=0, v=1, no_mode=no_mode)
    
    current = np.zeros(Gamma_R_range.size)
    #F2 = np.zeros(Gamma_R_range.size)
    voltage = np.zeros(Gamma_R_range.size)
    
    for i,G in enumerate(Gamma_R_range):
        print(G)
        model.Gamma_R = G
        H = model.construct_heom_matrix()
        jump_op = model.jump_matrix()
        pops = model.dv_pops()
        try:
            solver = FCSSolver(H, jump_op, pops)
            current[i] = solver.mean()
            #F2[i] = solver.second_order_fano_factor()
            ss = extract_system_density_matrix(solver.ss, model.el_dim)
            voltage[i] = calculate_voltage(1.4, 0, model.T, ss[3,3], ss[4,4])
        except ArpackNoConvergence:
            print('Got an ArpackNoConvergence error')
        
    np.savez(f'data/heom_photocell_deltaE_{int(delta_E)}_J_{int(J)}{c_label}N{num_tiers}_no_mode_{no_mode}.npz', \
             delta_E=delta_E, J=J, c=c, current=current, voltage=voltage)
        
