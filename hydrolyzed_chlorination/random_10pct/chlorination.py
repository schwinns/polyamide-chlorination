# Chlorinate the PA membrane

import matplotlib.pyplot as plt
import numpy as np

import MDAnalysis as mda

from polym_analysis import *
from gromacs_topology import *


def random_C_chlorination(polym, N, seed=12345):
    '''Randomly select N of the aromatic carbon atoms for chlorination'''

    ca_bonded_n = polym.universe.select_atoms('type ca and bonded type n') # force it to only be linear MPD fragments
    tmp = polym.universe.select_atoms('type ca and bonded group tmp', tmp=ca_bonded_n)
    can_chlorinate = []
    for atom in tmp:
        n_bonded_n = len([a for a in atom.bonded_atoms if a in ca_bonded_n])
        if n_bonded_n == 2:
            can_chlorinate.append(atom)

    can_chlorinate = mda.AtomGroup(can_chlorinate)
    rng = np.random.default_rng(seed=seed)
    rand_idx = rng.choice(np.arange(0,len(can_chlorinate)), size=N, replace=False)
    
    return can_chlorinate[rand_idx]

if __name__ == '__main__':
    
    # load in the PA membrane
    path = '/Users/nate_schwindt/Projects/polymer_membrane/polyamide_XAS_MD/Na/NaCl_polymer/50pct_deprotonated/OPC3/placed_Na/'

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
    
    # load in the topology for the original PA
    top = GromacsTopology('./PA_ions.top', verbose=True)

    # randomly select hydrogens to change on aromatic carbons to Cl in top and gro
    ca_chlorinate = random_C_chlorination(poly, round(0.06*len(poly.polymer.select_atoms('element C'))))
    ha_chlorinate = poly.universe.select_atoms('type ha and bonded group ca_chlorinate', ca_chlorinate=ca_chlorinate)

    # write a new coordinate file with chlorinated atoms
    u = poly.universe.copy()

    new = u.atoms.names
    new[ha_chlorinate.indices] = 'Cl'
    u.add_TopologyAttr('names', new)

    new = u.atoms.types
    new[ha_chlorinate.indices] = 'cl'
    u.add_TopologyAttr('types', new)

    new = u.atoms.elements
    new[ha_chlorinate.indices] = 'Cl'
    u.add_TopologyAttr('elements', new)

    new = u.atoms.masses
    new[ha_chlorinate.indices] = 35.45
    u.add_TopologyAttr('masses', new)

    u.atoms.write('chlorinated.gro')

    # load in a fragment that has the new Cl parameters
    # NOTE: This fragment was parameterized with GAFF using moltemplate (to be consistent with the rest of the PA)
    #       Then we converted the moltemplate-generated files to Gromacs format, added a box, and generated a tpr
    param_gro = PolymAnalysis('1MPD_2TMC_4P_Cl_box.gro', frmt='GRO', tpr_file='1MPD_2TMC_4P_Cl_converted.tpr', 
                               xlink_c=xlink_c, xlink_n=xlink_n, term_n=term_n, cl_type=cl_type, 
                               ow_type=ow_type, hw_type=hw_type)
    param_top = GromacsTopology('1MPD_2TMC_4P_Cl_converted.top')

    # extract the parameters that change when the H is replaced with Cl
    cl = param_gro.universe.select_atoms('type cl') # indices in MDAnalysis universe are the same as in GromacsTopology
    cl_atom = param_top.atoms[cl.indices[0]]

    # bond, angle, and dihedral parameters
    ca_cl_bond = cl_atom.bonds[0]
    ca_ca_cl_angle = cl_atom.angles[0]
    for dih in cl_atom.dihedrals:
        # save each type of dihedral for reassignment
        if dih.type == 4:
            ca_ca_ca_cl_imp = dih # improper dihedral

        has_n = len([a for a in dih.atoms if a.type == 'n']) == 1
        if has_n:
            n_ca_ca_cl_dih = dih # N-C-C-Cl dihedral
            
        has_3ca = len([a for a in dih.atoms if a.type == 'ca']) == 3
        if has_3ca and dih.type == 1:
            ca_ca_ca_cl_dih = dih # C-C-C-Cl dihedral

    # reassign atom types, bonds, angles, dihedrals for chlorinated atoms
    idx = ha_chlorinate.indices
    for atom in top.atoms[idx]:

        # assign atom type
        atom.atomtype = top.atomtypes['cl']
        atom.type = 'cl'
        atom.atomname = 'Cl'
        atom.mass = 35.45

        # assign bond parameters
        molecule = atom.molecule
        bond = [b for b in molecule.bonds if atom in b.atoms][0] # only the ca-cl bond
        bond.type = ca_cl_bond.type
        bond._params = ca_cl_bond._params

        # assign angle parameters
        angle = [a for a in molecule.angles if atom in a.atoms][0]  # only the ca-ca-cl angle
        angle.type = ca_ca_cl_angle.type
        angle._params = ca_ca_cl_angle._params

        # assign dihedral parameters
        for dih in atom.dihedrals:
            if dih.type == 4: # change the improper dihedral
                dih.type = ca_ca_ca_cl_imp.type
                dih._params = ca_ca_ca_cl_imp._params
                continue

            has_n = len([a for a in dih.atoms if a.type == 'n']) == 1
            if has_n: # change the N-C-C-H to N-C-C-Cl dihedral
                dih.type = n_ca_ca_cl_dih.type
                dih._params = n_ca_ca_cl_dih._params
                continue

            has_3ca = len([a for a in dih.atoms if a.type == 'ca']) == 3
            if has_3ca and dih.type == 1: # change the C-C-C-H to C-C-C-Cl dihedral
                dih.type = ca_ca_ca_cl_dih.type
                dih._params = ca_ca_ca_cl_dih._params

    # write out the new topology file
    top.write('chlorinated.top')