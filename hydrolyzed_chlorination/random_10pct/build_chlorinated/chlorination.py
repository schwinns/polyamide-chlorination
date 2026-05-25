# Chlorinate the PA membrane

import matplotlib.pyplot as plt
import numpy as np
import os

import MDAnalysis as mda

from polym_analysis import *
from gromacs_topology import *

if __name__ == '__main__':
    
    # load in the PA membrane
    path = '/ocean/projects/chm230020p/schwinns/Projects/polymer_membrane/REACTER/Lx5_Ly5_Lz10/ion_dynamics/preinserted_20pct/50pct_deprotonated/NaCl_0.2M/OPC3/placed_Na/'

    xlink_c = 'c and bonded type n'
    xlink_n = 'n'
    term_n = 'nh'
    cl_type = 'cl'
    oh_type = 'oh'
    ow_type = 'OW'
    hw_type = 'HW'
    cation_type = 'Na'
    anion_type = 'Cl'

    poly = PolymAnalysis(path+'prod.gro', frmt='GRO', tpr_file=path+'prod.tpr',
                        xlink_c=xlink_c, xlink_n=xlink_n, term_n=term_n,
                        cl_type=cl_type, oh_type=oh_type, ow_type=ow_type, hw_type=hw_type,
                        cation_type=cation_type, anion_type=anion_type)

    # poly.gmx = lambda n: f'mpirun -np {n} gmx_mpi' 
    poly.gmx = lambda n: ['mpirun', '-np', str(n), 'gmx_mpi']
    poly.hydrolyzed_chlorination(target_crosslink_pct=0.8, radius_for_water_search=5, rng_seed=3, n_counterions=1,
                                 input_topology='PA_ions.top', param_fragment='2MPD_2TMC_1P_Cl', nproc=64, restart_iteration=95)
