import numpy as np
import scipy.constants as constants
import quant_mech.utils as utils
import quant_mech.time_utils as tutils
from photocell_model import PhotocellModel
from counting_statistics.sparse.fcs_solver import FCSSolver
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
import logging
import datetime
import multiprocessing as mp
import sys

start_time = datetime.datetime.now()
timestamp = int(start_time.timestamp())

logging.basicConfig(
            filename=f'log/{__file__.split(".")[0]}-{timestamp}.log',
            level=logging.DEBUG
        )
logging.info(f'Executing script {__file__} at {start_time}')

# get the max num of processes to use from command line arg 
try:
    num_processes = int(sys.argv[1])
except:
    num_processes = 0
cpu_count = mp.cpu_count()
logging.info(f'CPU count: {cpu_count}')

if not num_processes:
    num_processes = cpu_count - 1
elif num_processes > (cpu_count - 1):
    logging.warn('Max number of processes defined on command line greater than number of cpus available')
    num_processes = cpu_count - 1
logging.info(f'Number of processes to use: {num_processes}')


Gamma_R_range = np.logspace(
                        -20.,
                        5.,
                        num=20,
                        base=np.exp(1.)
                    )*utils.EV_TO_WAVENUMS


params = [
    (240., 120., 4.69, False), # eta = 1, asymmetric
    (240., 120., 4.69, True),  # eta = 1, asymmetric, no mode
    (340., 8.5, 10.72, False), # eta = 0.05, asymmetric
    (340., 8.5, 10.72, True),  # eta = 0.05, asymmetric, no mode
    (240., 120., 1, False),    # eta = 1, symmetric
    (240., 120., 1, True),     # eta = 1, symmetric, no mode
    (340., 8.5, 1, False),     # eta = 0.05, symmetric
    (340., 8.5, 1, True)       # eta = 0.05, symmetric, no mode
]

# create temporary data files
for param_set in params:
    np.savez(f'data/data-temp_deltaE-{param_set[0]}_J-{param_set[1]}_c-{param_set[2]}_noMode-{param_set[3]}_{timestamp}.npz', current=np.zeros(Gamma_R_range.size), voltage=np.zeros(Gamma_R_range.size))

# function to call in process pool
def calculate_current(Gamma_R, i, delta_E, J, c, no_mode):

    def calculate_voltage(alpha_energy, beta_energy, temperature,
                            alpha_pop, beta_pop):
        return ((alpha_energy-beta_energy)*utils.EV_TO_JOULES + (constants.k * temperature)*np.log(alpha_pop/beta_pop)) / constants.e

    def extract_system_density_matrix(hierarchy_vector, sys_dim):
        sys_dm = hierarchy_vector[:sys_dim**2]
        sys_dm.shape = sys_dim, sys_dim
        return sys_dm

    p_name = mp.current_process().name
    logging.info(f'{p_name}: Starting task with Gamma_R = {Gamma_R} at {datetime.datetime.now().time()}')

    drude_reorg_energy = 35.
    drude_cutoff = 40.
    mode_freq = 342.
    mode_S = 0.0438
    mode_damping = 10.
    mode_reorg_energy = mode_freq * mode_S
    total_reorg_energy = drude_reorg_energy + mode_reorg_energy
    T = 300.
    num_tiers = 5

    logging.info(f'{p_name}: Constructing model at {datetime.datetime.now().time()}')
    model = PhotocellModel(
                    delta_E, J,
                    drude_reorg_energy, drude_cutoff,
                    mode_freq, mode_S, mode_damping,
                    c, temperature=T,
                    N=num_tiers, K=0, v=1, no_mode=no_mode
                )
    model.Gamma_R = Gamma_R
    H = model.construct_heom_matrix()
    jump_op = model.jump_matrix()
    pops = model.dv_pops()
    try:
        logging.info(f'{p_name}: Starting current calculation at {datetime.datetime.now().time()}')
        solver = FCSSolver(H, jump_op, pops)
        current = solver.mean()
        ss = extract_system_density_matrix(solver.ss, model.el_dim)
        voltage = calculate_voltage(1.4, 0, model.T, ss[3,3], ss[4,4])
    except ArpackNoConvergence:
        logging.error(f'{p_name}: Got an ArpackNoConvergence error at {datetime.datetime.now().time()}')
        logging.info(f'{p_name}: Completing task...')
        return 0, 0

    # save result
    logging.info(f'{p_name}: Calculation complete at {datetime.datetime.now().time()}')
    logging.info(f'{p_name}: Saving data...')
    try:
        temp_data = np.load(f'data/data-temp_deltaE-{delta_E}_J-{J}_c-{c}_noMode-{no_mode}_{timestamp}.npz')
        current_vals = temp_data['current']
        current_vals[i] = current
        voltage_vals = temp_data['voltage']
        voltage_vals[i] = voltage
        np.savez(f'data/data-temp_deltaE-{delta_E}_J-{J}_c-{c}_noMode-{no_mode}_{timestamp}.npz',
                 current=current_vals,
                 voltage=voltage_vals
        )
    except:
        logging.exception(f'{p_name}: Could not save current and voltage for Gamma_R = {Gamma_R}.')
        logging.info(f'{p_name}: Values were current {current} and voltage {voltage}.')

    return current, voltage

if __name__ == '__main__':
    #num_cpus = mp.cpu_count()
    #logging.info(f'Number of CPUs: {num_cpus}')
    
    for param_set in params:
        logging.info('#################################################')
        logging.info(f'Starting process pool with parameter set {param_set}')
        logging.info(f'Time is {datetime.datetime.now()}')
        logging.info('#################################################')
        with mp.Pool(num_processes) as pool:
            delta_E, J, c, no_mode = param_set
            args = [[Gamma_R, i, delta_E, J, c, no_mode] for i,Gamma_R in enumerate(Gamma_R_range)]
            result = pool.starmap(calculate_current, args)

            # save results all in one place
            logging.info(f'Processing complete at {datetime.datetime.now()}')
            logging.info(f'Saving results...')
            current, voltage = map(np.array, zip(*result))
            np.savez(f'data/data-complete_deltaE-{delta_E}_J-{J}_c-{c}_noMode-{no_mode}_{timestamp}.npz',
                     current=current,
                     voltage=voltage,
                     Gamma_R=Gamma_R_range
            )

logging.info('Execution complete!')

