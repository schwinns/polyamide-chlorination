# Script to reassign charges to polymer

from polym_analysis import PolymAnalysis
from gromacs_topology import *
from monomer_classes import *
import numpy as np
import yaml
from tqdm import tqdm
import subprocess
import argparse

def run(commands, cwd=None):
    '''Run commands with subprocess'''
    if not isinstance(commands, list):
        commands = [commands]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True, cwd=cwd)


def grompp(gro, mdp, top, tpr=None, gmx='gmx', flags={}, dry_run=False, **kwargs):
    '''
    Run grompp with mdp file on gro file with topology top
    
    flags should be a dictionary containing any additional flags, e.g. flags = {'maxwarn' : 1}
    '''
    if tpr is None:
        tpr = gro.split('.gro')[0] + '.tpr'
    cmd = [f'{gmx} grompp -f {mdp} -p {top} -c {gro} -o {tpr}']
    
    for f in flags:
        cmd[0] += f' -{f} {flags[f]}'

    if dry_run:
        print(cmd)
    else:
        run(cmd, **kwargs)
    return tpr


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coords', type=str, help='Coordinate file to reassign charges')
    parser.add_argument('-p', '--top', type=str, help='Topology file to reassign charges')
    parser.add_argument('-s', '--tpr', type=str, help='TPR file to reassign charges')
    args = parser.parse_args()

    coordinates = args.coords
    topology = args.top
    tpr = args.tpr

    # define proper types for Gromacs system
    xlink_c = 'c and bonded type n'
    xlink_n = 'n'
    term_n = 'nh'
    cl_type = 'cl'
    oh_type = 'oh'
    ow_type = 'OW'
    hw_type = 'HW'
    cation_type = 'Na'
    anion_type = 'Cl'

    gro = PolymAnalysis(coordinates, frmt='GRO', tpr_file=tpr,
                        xlink_c=xlink_c, xlink_n=xlink_n, term_n=term_n,
                        cl_type=cl_type, oh_type=oh_type, ow_type=ow_type, hw_type=hw_type,
                        cation_type=cation_type, anion_type=anion_type)

    # initialize GromacsTopology
    top = GromacsTopology(topology, verbose=True)

    # reassign charges to whole polymer
    atoms = gro.universe.select_atoms('resname PA*')
    original_charges = atoms.charges

    with open('charges_modified.yaml', 'r') as file: # read in charges from yaml file
        charges_dict = yaml.safe_load(file)

    print('Creating MPD monomer fragments...')
    mpds = []
    Ns = []
    for N in tqdm(atoms.select_atoms('type n nh')):

        has_cl = 'cl' in N.bonded_atoms.types

        if not has_cl: # check other N
            my_ca = [atom for atom in N.bonded_atoms if atom.type == 'ca'][0] # get connected aromatic C
            next_ca = [atom for atom in my_ca.bonded_atoms if atom.type == 'ca'] # get next aromatic C
            for ca in next_ca:
                next_next_ca = [atom for atom in ca.bonded_atoms if atom != my_ca and atom.type == 'ca'][0]
                if 'N' in next_next_ca.bonded_atoms.elements:
                    n = [atom for atom in next_next_ca.bonded_atoms if atom.element == 'N'][0]
                    break

            has_cl = 'cl' in n.bonded_atoms.types

        if has_cl:
            mpd = MPD_Cl_T(N, cl='cl', xlink_n='n', ar_c='ca', term_n='nh', ar_h='ha', hn='hn')
        else:
            mpd = MPD(N, xlink_n='n', ar_c='ca', term_n='nh', ar_h='ha', hn='hn')

        if mpd.N2 not in Ns:
            mpd.assign_charges(charges_dict[mpd.name])
            mpds.append(mpd)
        
        Ns.append(mpd.N1)
        Ns.append(mpd.N2)

    print('Creating TMC monomer fragments...')
    tmcs = []
    Cs = []
    for C in tqdm(atoms.select_atoms('type c')):
        tmc = TMC(C, ar_c='ca', xlink_c='c', ar_h='ha', deprot_o='o', prot_o='oh', ho_type='ho')
        if tmc.C2 not in Cs and tmc.C3 not in Cs:
            tmc.assign_charges(charges_dict[tmc.name])
            tmcs.append(tmc)

        Cs.append(tmc.C1)
        Cs.append(tmc.C2)
        Cs.append(tmc.C3)
        
    for atom in atoms: # assign new charges to top
        idx = atom.index
        top_atom = top.atoms[idx]
        top_atom.charge = atom.charge

    new_charges = atoms.charges
    print('\nTotal charge before reassignment: {:.4f}'.format(original_charges.sum()))
    print('Total charge after reassignment: {:.4f}'.format(new_charges.sum()))
    
    # write new topology
    top.write('recharged.top')