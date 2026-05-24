# Class for building and analysis of polyamide membrane

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
import subprocess
import yaml

import networkx as nx

import MDAnalysis as mda
import MDAnalysis.transformations as trans
from MDAnalysis.analysis import distances


def _run(cmd, cwd=None):
    '''Run commands with subprocess'''
    return subprocess.run(cmd, cwd=cwd, check=True)


class PolymAnalysis():
    def __init__(self, data_file, frmt='DATA', init_file='system.in.init', settings_file='cleanedsystem.in.settings',
                 tpr_file=None, xlink_c='1', xlink_n='7', term_n='8', cl_type='6', oh_type='10',
                 ow_type='11', hw_type='12', cation_type='', anion_type=''):

        self.lmp = '23' # extension for the LAMMPS executable
        self.gmx = 'gmx' # name of the GROMACS executable

        # Save information about original file
        print('Initializing polyamide membrane for analysis...')
        self.filename = data_file
        self.ext = data_file.split('.')[-1]

        self.init_file = init_file
        self.settings_file = settings_file

        # Create an internal MDAnalysis Universe and atoms AtomGroup
        if tpr_file is None:
            self.universe = mda.Universe(data_file, format=frmt)
            self.tpr = None
        else:
            self.universe = mda.Universe(tpr_file, data_file)
            self.tpr = tpr_file
        self.atoms = self.universe.atoms

        if not self.ext in ['xtc', 'trr']:
            f = open(data_file, 'r')
            self.original = f.readlines()
            self.trajectory = None
        else:
            self.trajectory = self.universe.trajectory

        # Save the monomer types and prepare space for log files
        self.monomers = ['MPD', 'TMC']
        self.logs = []

        # Get information about number of atoms and monomers
        if self.ext == 'data' or tpr_file is not None:
            self.atsel = 'type'
        elif self.ext == 'gro':
            self.atsel = 'name'

        xlink_c_ag = self.universe.select_atoms('{} {}'.format(self.atsel, xlink_c))
        xlink_n_ag = self.universe.select_atoms('{} {}'.format(self.atsel, xlink_n), xlink_c=xlink_c_ag)
        term_n_ag = self.universe.select_atoms('{} {}'.format(self.atsel, term_n), xlink_c=xlink_c_ag)

        self.n_atoms = len(self.atoms)
        nMPD = int((len(xlink_n_ag) + len(term_n_ag)) / 2)
        nTMC = int(len(xlink_c_ag) / 3)
        nH2O = int(len(self.universe.select_atoms('{} {}'.format(self.atsel, ow_type))))
        self.n_monomers = [nMPD, nTMC]
        print(f'PA membrane consists of {nMPD} MPD monomers, {nTMC} TMC monomers, {nH2O} waters for a total of {self.n_atoms} atoms')

        # Store input information about atom types
        self.xlink_n_type = xlink_n
        self.xlink_c_type = xlink_c
        self.term_n_type = term_n
        self.cl_type = cl_type
        self.oh_type = oh_type
        self.ow_type = ow_type
        self.hw_type = hw_type
        self.cation_type = cation_type
        self.anion_type = anion_type

        # Get coordinate information
        self.coords = np.zeros((1,self.n_atoms, 3))
        self.coords[0,:,:] = self.atoms.positions
        self.coords_lims = np.zeros((2,3))
        # self.coords_lims[0,:] = self.atoms.positions.min(axis=1)
        self.coords_lims[:,0] = self.atoms.positions[:,0].min(), self.atoms.positions[:,0].max()
        self.coords_lims[:,1] = self.atoms.positions[:,1].min(), self.atoms.positions[:,1].max()
        self.coords_lims[:,2] = self.atoms.positions[:,2].min(), self.atoms.positions[:,2].max()

        # Save original charges and box size
        self.charges = self.atoms.charges
        self.box = self._get_box()

        # Get header information
        if self.ext == 'data':
            self.masses = self._get_masses(data_file)
            self._get_topology_types(data_file)
            self.density = self.calculate_density(box=True)

        print('Finished loading PA membrane for analysis!\n')


    def __str__(self):
        return f'<PolymAnalysis object with {self.n_atoms} atoms from {self.filename}>'

    @property
    def polymer(self):
       return self.universe.select_atoms(f'not type {self.ow_type} {self.hw_type} {self.anion_type} {self.cation_type}')

    def _run(self, commands, cwd=None):
        '''Run commands with subprocess'''
        if not isinstance(commands, list):
            commands = [commands]
        
        for cmd in commands:
            subprocess.run(cmd, shell=True, cwd=cwd, check=True)

    
    def _get_topology_types(self, file):
        '''Get topology numbers and types from a LAMMPS data file'''
        f = open(file, 'r')
        lines = f.readlines()

        for line in lines[1:]: # always skip first line

            kws = ['atoms', 'bonds', 'angles', 'dihedrals', 'impropers', 'atom types', 'bond types', 'angle types', 'dihedral types', 'improper types']
            for kw in kws:
                if kw in line:
                    setattr(self, f"n_{kw.replace(' ', '_')}", int(line.split()[0])) 


    def _get_box(self):
        '''Get box limits from data file'''

        box = np.zeros((2,3))
        
        if self.ext == 'data':
            f = open(self.filename, 'r')
            lines = f.readlines()
            for line in lines:
                if 'xlo' in line:
                    box[0,0] = float(line.split()[0])
                    box[1,0] = float(line.split()[1])
                elif 'ylo' in line:
                    box[0,1] = float(line.split()[0])
                    box[1,1] = float(line.split()[1])
                elif 'zlo' in line:
                    box[0,2] = float(line.split()[0])
                    box[1,2] = float(line.split()[1])

        elif self.ext in ['gro', 'xtc', 'trr']:
            box[1,:] = self.universe.dimensions[:3]

        return box
    

    def _get_masses(self, file):
        '''Get atom types and masses from a LAMMPS data file'''
        f = open(file, 'r')
        lines = f.readlines()

        # Get indices of where Masses starts
        idx1 = lines.index('Masses\n')

        # Loop through masses and save in dictionary (masses[atom_type] = mass)
        masses = {}
        for line in lines[idx1+2:]:

            if len(line.split()) == 0:
                break
            else:
                atom_type = int(line.split()[0])
                mass = float(line.split()[1])
                masses[atom_type] = mass

        return masses
    
    
    def _reassign_atom_numbers(self, output='renumbered.data'):
        '''Reassign atom indices to be in numerical order'''

        from lammps import PyLammps

        # Use LAMMPS to reset ids
        L = PyLammps(self.lmp)
        L.file(self.init_file)
        L.read_data(self.filename)
        L.file(self.settings_file)

        L.reset_atom_ids()
        L.reset_mol_ids('all')

        # write the new system to data file and reinitialize object
        L.write_data(output)
        print()
        self.__init__(output)

    
    def _read_init(self, file=None):
        '''Get styles and settings from the init file (as generated by moltemplate)'''

        if file is None: # use init file from instantiation
            file = self.init_file

        f = open(file, 'r')
        lines = f.readlines()

        # create a Settings class to save information
        self.init_settings = Settings()
        
        for line in lines:
            l = line.split()
            # if line is commented out or blank ignore
            if len(l) == 0:
                pass
            elif not l[0].startswith('#'):
                setattr(self.init_settings, l[0], l[1:])


    def _read_settings(self, file=None):
        '''Get pair, bond, angle, etc. coefficients from the settings file (as generated by moltemplate)'''

        if file is None: # use settings file from instantiation
            file = self.settings_file

        f = open(file, 'r')
        lines = f.readlines()

        # create Settings class to save information
        self.coeff_settings = Settings()

        self.coeff_settings.pair_coeff = {}
        self.coeff_settings.bond_coeff = {}
        self.coeff_settings.angle_coeff = {}
        self.coeff_settings.dihedral_coeff = {}
        self.coeff_settings.improper_coeff = {}

        print(f'Reading coefficients from {file}...')
        print('WARNING: can only read the following coefficients:')
        print('\tpair_coeff: [LJ epsilon and sigma]')
        print('\tbond_coeff: [harmonic]')
        print('\tangle_coeff: [harmonic]')
        print('\tdihedral_coeff: [fourier]')
        print('\timproper_coeff: [cvff]')

        for line in lines:
            l = line.split('#')[0].split()

            if len(l) == 0:
                pass
            elif l[0].startswith('#'):
                pass
            elif l[0] == 'pair_coeff':
                type1 = int(l[1])
                type2 = int(l[2])
                style = l[3]
                eps = float(l[4])
                sig = float(l[5])

                if type1 == type2:
                    self.coeff_settings.pair_coeff[type1] = {
                            'style' : style,
                            'epsilon' : eps,
                            'sigma' : sig     # for lj/charmm/coul/long this is the "zero-crossing distance"
                        }
                else:
                    self.coeff_settings.pair_coeff[f'{type1}-{type2}'] = {
                            'style' : style,
                            'epsilon' : eps,
                            'sigma' : sig     # for lj/charmm/coul/long this is the "zero-crossing distance"
                        }
                
            elif l[0] == 'bond_coeff':
                b = int(l[1])
                style = l[2]
                K = float(l[3])
                r = float(l[4])

                self.coeff_settings.bond_coeff[b] = {
                    'style' : style,
                    'K' : K, # kcal/mol/A^2
                    'r0' : r # Angstroms
                }

            elif l[0] == 'angle_coeff':
                ang = int(l[1])
                style = l[2]
                K = float(l[3])
                theta = float(l[4])

                self.coeff_settings.angle_coeff[ang] = {
                    'style' : style,
                    'K' : K, # kcal/mol
                    'theta0' : theta # degrees
                }

            elif l[0] == 'dihedral_coeff':
                dih = int(l[1])
                style = l[2]
                m = int(l[3])
                K = l[4::3]
                n = l[5::3] 
                d = l[6::3]

                self.coeff_settings.dihedral_coeff[dih] = {
                    'style' : style,
                    'm' : m,
                    'K' : np.array([float(i) for i in K]), # kcal/mol
                    'n' : np.array([int(i) for i in n]),
                    'd' : np.array([float(i) for i in d]) # degrees
                }

            elif l[0] == 'improper_coeff':
                imp = int(l[1])
                style = l[2]
                K = float(l[3])
                d = int(l[4])
                n = int(l[5])

                self.coeff_settings.improper_coeff[imp] = {
                    'style' : style,
                    'K' : K, # kcal/mol
                    'd' : d,
                    'n' : n
                }


    def _atomtype_mapping(self, map):
        '''Provide a mapping from integer atom types to alphanumeric atom types'''
        self.type_map = map # TODO: add some checks or functionality here 


    def _guess_elements(self, masses=None):
        '''Guess elements based on masses'''

        if masses is None:
            masses = self.masses

        self.element_map = {}
        for a in masses:
            mass = masses[a]
            self.element_map[a] = {}
            if mass >= 12 and mass <= 12.02:
                self.element_map[a]['name'] = 'C'
                self.element_map[a]['number'] = 6
            elif mass >= 14 and mass <= 14.01:
                self.element_map[a]['name'] = 'N'
                self.element_map[a]['number'] = 7
            elif mass >= 35 and mass <= 35.5:
                self.element_map[a]['name'] = 'Cl'
                self.element_map[a]['number'] = 17
            elif mass >= 15.9 and mass <= 16.1:
                self.element_map[a]['name'] = 'O'
                self.element_map[a]['number'] = 8
            elif mass >= 1 and mass <= 1.008:
                self.element_map[a]['name'] = 'H'
                self.element_map[a]['number'] = 1
            else:
                raise TypeError(f'Element {a} with mass {mass} not implemented... Add it to self._guess_elements')

    def _generate_tpr(self, top, coord, mdp='min.mdp', tpr=None, flags={}, gmx='gmx'):
        '''Use Gromacs gmx grompp to generate a tpr file'''

        if tpr is None:
            tpr = coord.split('.')[0] + '.tpr'
        
        # Build argv list safely. .extend() returns None so don't assign its result.
        if callable(gmx):
            gmx_argv = list(gmx(1))  # expect callable to return a list-like argv
        elif isinstance(gmx, (list, tuple)):
            gmx_argv = gmx
        else:
            raise TypeError('gmx must be a callable or list/tuple of command arguments')

        cmd = gmx_argv + ['grompp', '-f', mdp, '-p', top, '-c', coord, '-o', tpr]
        # cmd = [f'{gmx} grompp -f {mdp} -p {top} -c {coord} -o {tpr}']

        # append flags (dict) as ["-flag", "value"]
        # for key, val in flags.items():
        #     cmd[0] += f' -{key} {val}'
        for key, val in flags.items():
            cmd += [f'-{key}', f'{val}']

        _run(cmd)
        return tpr


    def load_logs(self, log_files):
        '''Load LAMMPS log files into PolymAnalysis class'''

        import lammps_logfile

        if not isinstance(log_files, list):
            log_files = [log_files]

        for log_file in log_files:
            log = lammps_logfile.File(log_file)
            self.logs.append(log)
        
        print('Currently loaded log files:')
        [print(log.logfile.name) for log in self.logs]


    def load_trajectory(self, traj_file, center=False, save_as_array=False):
        '''Load trajectory data into PolymAnalysis class'''

        print('Loading all trajectory coordinates from {}'.format(traj_file))
        self.trajectory_file = traj_file

        if self.tpr is None:
            self.universe = mda.Universe(self.filename, traj_file)
        else:
            self.universe = mda.Universe(self.tpr, traj_file)

        self.trajectory = self.universe.trajectory
        self.n_frames = len(self.trajectory)

        workflow = []
        if center:
            workflow.append(trans.unwrap(self.universe.atoms))
            workflow.append(trans.center_in_box(self.polymer, center='mass'))
            workflow.append(trans.wrap(self.universe.atoms))
            self.universe.trajectory.add_transformations(*workflow)


        if save_as_array:
            self.coords = np.zeros((self.n_frames, self.n_atoms, 3))
            
            t = 0
            for ts in tqdm(self.trajectory):
                self.coords[t,:,:] = self.universe.atoms.positions
                t += 1


    def create_polymer_graphs(self):
        '''Create a graph representation of the polymer network'''

        self.polymer_graphs = []

        for res in self.polymer.residues:
            g = build_graph_from_atoms(res.atoms)
            self.polymer_graphs.append(g)

        return self.polymer_graphs


    def pack_water_reservoirs(self, output='prehydrate.data', water_file='water.data', box_frac=0.5, seed=12345):
        '''Pack two water reservoirs on either side of the PA slab'''

        from lammps import PyLammps

        # Calculate number of water molecules for each reservoir
        wat_mass = 16 + 1.008*2
        box = self.coords_lims
        s = (box[1,:] - box[0,:])
        s[2] = s[2]*box_frac
        vol = s[0]*s[1]*s[2] / 10**3
        n_waters = int(vol * 6.022 / wat_mass * 10**2) # assuming 1 g/cm^3 water density
        print(f'Adding {n_waters} waters to each reservoir...')

        # Find new box limits
        ub = box[1,2] + s[2]
        lb = box[0,2] - s[2]

        # Use LAMMPS to add water reservoirs to either side and write data file
        L = PyLammps(self.lmp)
        L.file(self.init_file)

        # add topology types if not there already
        if len(self.universe.select_atoms(f'{self.atsel} {self.ow_type}')) == 0:
            L.read_data(self.filename, 'extra/atom/types', 2, 'extra/bond/types', 1, 'extra/angle/types', 1)
            L.mass(int(self.ow_type), 16)
            L.mass(int(self.hw_type), 1.008)
        else:
            L.read_data(self.filename)

        L.file(self.settings_file)

        # convert to fully periodic
        L.change_box('all', 'z', 'final', lb, ub) # add space for reservoirs
        L.change_box('all', 'boundary', 'p p p') # make box fully periodic

        # create regions to fill with waters
        L.region('top block', box[0,0], box[1,0], box[0,1], box[1,1], box[1,2], ub)
        L.region('bott block', box[0,0], box[1,0], box[0,1], box[1,1], lb, box[0,2])
        
        # read molecule template for water (from https://docs.lammps.org/Howto_tip3p.html)
        L.molecule('water', water_file)

        # create water molecules in reservoirs
        L.create_atoms(int(self.ow_type)-1, 'random', n_waters, seed, 'top', 'mol water', seed+1, 'overlap', 1.33)
        L.create_atoms(int(self.ow_type)-1, 'random', n_waters, seed+2, 'bott', 'mol water', seed+3, 'overlap', 1.33)

        # write the new system to data file and reinitialize object
        L.write_data(output)
        self.__init__(output)


    def pack_term_groups(self, output='packed.data', rad=10, n_OH=5, OH_file='OH.data', seed=12345):
        '''Pack free OH groups near unterminated Cl'''

        from lammps import PyLammps

        # Use LAMMPS to add free OH groups
        L = PyLammps(self.lmp)
        L.file(self.init_file)
        L.read_data(self.filename)
        L.file(self.settings_file)

        Cl_group = self.atoms.select_atoms(f'{self.atsel} {self.cl_type}')
        print(f'Trying to insert {len(Cl_group)*n_OH} OH groups ({len(Cl_group)*n_OH*2} atoms) around {len(Cl_group)} unterminated Cl for a total of {self.n_atoms + len(Cl_group)*n_OH*2} atoms in system...')
        L.molecule('OH', OH_file)

        for i,Cl in enumerate(Cl_group):

            x, y, z = Cl.position

            # create sphere at the Cl with radius rad and place n_OH terminating groups within sphere
            L.region(f'sph{i}', 'sphere', x, y, z, rad, 'side in')

            # create free OH molecules in region
            L.create_atoms(0, 'random', n_OH, seed, f'sph{i}', 'mol OH', seed+1, 'overlap', 1.33)

        # write the new system to data file and reinitialize object
        L.reset_atom_ids()
        L.reset_mol_ids('all')        
        L.write_data(output)
        self.__init__(output)

        print(f'Final number of atoms: {self.n_atoms}')


    def set_artifical_charges(self, atom_group1, atom_group2, charge1=0.5, charge2=-0.5):
        '''Set artificial charges on reacting groups atom_group1 and atom_group2 and 0 charge on all other atoms'''

        if isinstance(atom_group1, str): # if provided selection language, make AtomGroup
            g1 = self.universe.select_atoms(atom_group1)
        else: # else assume input is AtomGroup
            g1 = atom_group1

        if isinstance(atom_group2, str):
            g2 = self.universe.select_atoms(atom_group2)
        else:
            g2 = atom_group2

        others = self.universe.select_atoms('not group g1 and not group g2', g1=g1, g2=g2)

        print('Total charge in system is originally: {:.4f}'.format(self.charges.sum()))

        self.charges[g1.indices] = charge1
        self.charges[g2.indices] = charge2
        self.charges[others.indices] = 0

        print('Total charge in system is now: {:.4f}'.format(self.charges.sum()))


    def write_ndx(self, atom_groups, output='index.ndx', group_names=None):
        '''Write a Gromacs-style index file that associates atom IDs with the corresponding groups (can be used with ndx2group command in LAMMPS to create group)'''

        if not isinstance(atom_groups, list):
            raise TypeError('Input atom_groups as a list of atom_groups')
        
        groups = []
        for atom_group in atom_groups:
            if isinstance(atom_group, str): # if provided selection language, make AtomGroup
                groups.append(self.universe.select_atoms(atom_group))
            else: # else assume input is AtomGroup
                groups.append(atom_group)

        self.groups = groups
        self.ndx = output
        out = open(output, 'w')
        for g, group in enumerate(groups): 
            if group_names is not None: # use names if provided
                name = group_names[g]
            else:
                name = f'Group{g}'

            out.write(f'[ {name} ]\n') # write the group name
            for i, idx in enumerate(group.indices+1): # loop through all indices and go to next line every 12
                if (i+1) % 12 == 0:
                    out.write(f'{idx}\n')
                else:
                    out.write(f'{idx} ')


    def write_GROMACS(self, output='PA', water_itp='HOH.itp'):
        '''Write topology and coordinate information to Gromacs top and gro files'''

        from textwrap import dedent

        self._read_init()
        self._read_settings()
        self._guess_elements()

        #### WRITE TOP FILE ####

        top = open(output + '.top', 'w')

        # [ defaults ]
        if self.init_settings.pair_modify[0] == 'mix' and self.init_settings.pair_modify[1] == 'arithmetic':
            comb_rule = 2
        else:
            raise TypeError(f'pair_modify {self.init_settings.pair_modify} has not been implemented yet')
    
        if self.init_settings.special_bonds[0] == 'amber':
            fudgeLJ = 0.5
            fudgeQQ = 0.8333
        else:
            print('No fudgeLJ and fudgeQQ specified. Setting them to 1.')
            fudgeLJ = 1
            fudgeQQ = 1

        defaults = dedent(f"""\
        [ defaults ]
        ; nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ
        \t1\t{comb_rule}\tyes\t{fudgeLJ}\t{fudgeQQ}\n
        """)
        top.write(defaults)

        # [ atomtypes ]
        atomtypes = dedent(f"""\
        [ atomtypes ]
        ;type, bondingtype, atomic_number, mass, charge, ptype, sigma, epsilon
        """)
        for a in self.masses:
            atype = self.type_map[a]
            at_num = self.element_map[a]['number']
            mass = self.masses[a]
            sig = self.coeff_settings.pair_coeff[a]['sigma'] / 10 # convert to nm
            eps = self.coeff_settings.pair_coeff[a]['epsilon'] * 4.184 # convert to kJ/mol

            line = f'{atype:4s} {atype:4s}\t{at_num:.0f}\t{mass:.8f}\t{0:.8f}\tA\t{sig:.8e}\t{eps:.8e}\n'
            atomtypes += line

        top.write(atomtypes)

        # [ moleculetype ] for each molecule in the system
        print('Writing topology information for polymer chains...')
        PA_chains = self.universe.select_atoms('not type 11 12')
        n_chains = len(PA_chains.residues)
        count_atoms = 0
        count_molecules = 0
        sorted_atoms = []
        resnames = np.empty(len(self.universe.residues), dtype=object)
        atomnames = np.empty(self.n_atoms, dtype=object)
        for m in tqdm(range(n_chains)):
            mol = self.universe.select_atoms('resid {}'.format(PA_chains.residues[m].resid))
            resnames[count_molecules] = f'PA{m+1}'
            count_molecules += 1

            molecule = dedent(f'''
            [ moleculetype ]
            PA{m+1}\t\t3

            [ atoms ]
            ;num, type, resnum, resname, atomname, cgnr, q, m
            ''')

            # [ atoms ] for each molecule
            types_per_res = {}
            atom_in_molecule = {}
            for a,atom in enumerate(mol.atoms):
                sorted_atoms.append(atom)
                atype = self.type_map[int(atom.type)]
                resnum = m+1
                resname = f'PA{m+1}'
                element = self.element_map[int(atom.type)]['name']
                charge = atom.charge
                mass = atom.mass
                atom_in_molecule[atom.id] = a+1

                # NOTE: TOO MANY ATOMS --> DO NOT NUMBER THEM WITHIN EACH MOLECULE
                # if element in types_per_res:
                #     types_per_res[element] += 1
                #     atomname = f'{element}{types_per_res[element]}'
                # else:
                #     types_per_res[element] = 0
                #     atomname = f'{element}'
                atomname = element

                atomnames[count_atoms] = atomname
                count_atoms += 1
                line = f'{a+1:7d} {atype:4s}\t\t{resnum} {resname:8s} {atomname:9s}\t0\t{charge:.8f}\t{mass:.8f}\n'
                molecule += line

            # [ bonds ] for each molecule
            molecule += dedent(f'''
            [ bonds ]
            ;   ai     aj funct  r               k
            ''')

            for bond in mol.bonds:
                a1 = atom_in_molecule[bond.atoms[0].id]
                a2 = atom_in_molecule[bond.atoms[1].id]
                btype = int(bond.type)
                style = self.coeff_settings.bond_coeff[btype]['style']

                if style == 'harmonic':
                    func = 1
                    K = self.coeff_settings.bond_coeff[btype]['K'] * 2 * 4.184 * 10**2 # convert to kJ/mol/nm^2
                    r = self.coeff_settings.bond_coeff[btype]['r0'] / 10 # convert to nm
                else:
                    raise NotImplementedError(f'Bond style {style} not implemented yet')
                
                line = f'{a1:7d} {a2:>7d} {func}\t{r:.8e}\t{K:.8e}\n'
                molecule += line


            # [ angles ] for each molecule
            molecule += dedent(f'''
            [ angles ]
            ;   ai     aj     ak     funct  theta    cth
            ''')

            for ang in mol.angles:
                a1 = atom_in_molecule[ang.atoms[0].id]
                a2 = atom_in_molecule[ang.atoms[1].id]
                a3 = atom_in_molecule[ang.atoms[2].id]
                angtype = int(ang.type)
                style = self.coeff_settings.angle_coeff[angtype]['style']

                if style == 'harmonic':
                    func = 1
                    K = self.coeff_settings.angle_coeff[angtype]['K'] * 2 * 4.184
                    theta = self.coeff_settings.angle_coeff[angtype]['theta0']
                else:
                    raise NotImplementedError(f'Angle style {style} not implemented yet')
                
                line = f'{a1:7d} {a2:>7d} {a3:>7d} {func}\t{theta:.8e}\t{K:.8e}\n'
                molecule += line
            
            # [ dihedrals ] for each molecule
            molecule += dedent(f'''
            [ dihedrals ]
            ;    i      j      k      l   func
            ''')

            for dih in mol.dihedrals: # add proper dihedrals first

                a1 = atom_in_molecule[dih.atoms[0].id]
                a2 = atom_in_molecule[dih.atoms[1].id]
                a3 = atom_in_molecule[dih.atoms[2].id]
                a4 = atom_in_molecule[dih.atoms[3].id]
                dihtype = int(dih.type)
                style = self.coeff_settings.dihedral_coeff[dihtype]['style']

                if style == 'fourier' and self.coeff_settings.dihedral_coeff[dihtype]['m'] == 1: # if only one fourier term, this is a periodic proper dihedral
                    func = 1
                    K = self.coeff_settings.dihedral_coeff[dihtype]['K'][0] * 4.184
                    n = self.coeff_settings.dihedral_coeff[dihtype]['n'][0]
                    d = self.coeff_settings.dihedral_coeff[dihtype]['d'][0]
                    params = f'{d:.8e}\t{K:.8e}\t{n}'

                elif style == 'fourier': # otherwise it is a Fourier dihedral
                    func = 5
                    K = self.coeff_settings.dihedral_coeff[dihtype]['K'] * 4.184
                    n = self.coeff_settings.dihedral_coeff[dihtype]['n']
                    d = self.coeff_settings.dihedral_coeff[dihtype]['d']
                    C = np.zeros(4)

                    for Ki, ni, di in zip(K,n,d):
                        if ni == 1 and di == 0:
                            C[0] = 2*Ki
                        elif ni == 2 and di == 180:
                            C[1] = 2*Ki
                        elif ni == 3 and di == 0:
                            C[2] = 2*Ki
                        elif ni == 4 and di == 180:
                            C[3] = 2*Ki
                        else:
                            raise TypeError(f'Parameters {[Ki, ni, di]} cannot be converted to Gromacs dihedral style {style}')

                    params = f'{C[0]:.8e}\t{C[1]:.8e}\t{C[2]:.8e}\t{C[3]:.8e}'

                else:
                    raise NotImplementedError(f'Dihedral style {style} not implemented yet')
                
                line = f'{a1:7d} {a2:>7d} {a3:>7d} {a4:>7d} {func}\t{params}\n'
                molecule += line

            for imp in mol.impropers: # add improper dihedrals

                a1 = atom_in_molecule[imp.atoms[0].id]
                a2 = atom_in_molecule[imp.atoms[1].id]
                a3 = atom_in_molecule[imp.atoms[2].id]
                a4 = atom_in_molecule[imp.atoms[3].id]
                imptype = int(imp.type)
                style = self.coeff_settings.improper_coeff[imptype]['style']

                if style == 'cvff':
                    func = 4
                    K = self.coeff_settings.improper_coeff[imptype]['K'] * 4.184
                    n = self.coeff_settings.improper_coeff[imptype]['n']
                    d = self.coeff_settings.improper_coeff[imptype]['d']
                else:
                    raise NotImplementedError(f'Improper style {style} not implemented yet')
                
                line = f'{a1:7d} {a2:>7d} {a3:>7d} {a4:>7d} {func}\t{d:.8e}\t{K:.8e}\t{n}\n'
                molecule += line

            top.write(molecule)

        # include water itp if waters present
        OW = self.universe.select_atoms('type {}'.format(self.ow_type))
        n_waters = len(OW)
        if n_waters > 0:
            water = f'\n#include "{water_itp}"'
            top.write(water)

        # [ system ] and [ molecules ]
        system = dedent(f'''
        [ system ]
        {self.filename} converted

        [ molecules ]
        ; Compound        nmols
        ''')

        for n in range(count_molecules):
            line = f'PA{n+1}\t1\n'
            system += line

        if n_waters > 0:
            system += f'SOL\t{n_waters}\n'

        top.write(system)
        top.close()
        print(f'Finished writing Gromacs topology file: {output}.top!')

        #### WRITE GRO FILE ####
        print(f'Writing coordinate information...')

        for ow in OW: # now add waters
            wat = ow.residue
            hw1 = wat.atoms[1]
            hw2 = wat.atoms[2]

            if hw1.type != self.hw_type:
                raise TypeError(f'Atom {hw1} not a water hydrogen')
            elif hw2.type != self.hw_type:
                raise TypeError(f'Atom {hw2} not a water hydrogen')
            
            sorted_atoms.append(ow)
            sorted_atoms.append(hw1)
            sorted_atoms.append(hw2)
            
            resname = 'SOL'
            
            resnames[count_molecules] = resname
            atomnames[count_atoms] = 'OW'
            count_atoms += 1

            atomnames[count_atoms] = 'HW1'
            count_atoms += 1

            atomnames[count_atoms] = 'HW2'
            count_atoms += 1
            count_molecules += 1

        sorted_ag = mda.AtomGroup(sorted_atoms)
        self.universe.add_TopologyAttr('resnames')
        self.universe.add_TopologyAttr('names')
        sorted_ag.residues.resnames = resnames
        sorted_ag.atoms.names = atomnames
        sorted_ag.write(output + '.gro')

        print(f'Finished writing Gromacs coordinate file: {output}.gro!\n')


    def remove_atoms(self, atom_group, output='removed_atoms.data'): 
        '''Remove atoms in atom_group from the system'''
        
        from lammps import PyLammps

        if isinstance(atom_group, str): # if provided selection language, make AtomGroup
            g1 = self.universe.select_atoms(atom_group)
        else: # else assume input is AtomGroup
            g1 = atom_group

        # create a string with atom ids for group definition (not very clean...)
        ids = str(g1.indices[0] + 1)
        for idx in (g1.indices + 1)[1:]:
            ids += ' ' + str(idx)

        print(f'Deleting {len(g1)} atoms...')

        # Use LAMMPS to remove atoms and reset ids
        L = PyLammps(self.lmp)
        L.file(self.init_file)
        L.read_data(self.filename)
        L.file(self.settings_file)

        L.group('del', 'id', ids)
        L.delete_atoms('group', 'del')
        L.reset_atom_ids()
        L.reset_mol_ids('all')

        # write the new system to data file and reinitialize object
        L.write_data(output)
        print()
        self.__init__(output)


    def insert_cations_in_membrane(self, ion_name='Na', ion_charge=1, extra_inserted=0, tol=2, output='PA_ions.gro'):
        '''Add ions to the membrane by merging universe with ion universe'''

        # locate COOH/COO- oxygens
        c_group = self.universe.select_atoms('type c and not bonded type n')
        deprot_o = []
        prot_o = []
        for c in c_group:
            my_Os = [atom for atom in c.bonded_atoms if atom.type == 'o']
            if len(my_Os) == 2:
                deprot_o.append(my_Os[0]) # add one of the two O's to the AtomGroup
            elif len(my_Os) == 1:
                prot_o.append(my_Os[0]) # add the =O to the AtomGroup
            else:
                raise TypeError(f'{c} is not the carbon in R-COOH or R-COO-')
        
        # calculate excess charge from adding ions
        polymer_charge = len(deprot_o)*-1
        if len(deprot_o) % ion_charge > 0: # if we cannot exactly balance polymer charge with cations
            n_ions = round(len(deprot_o) / ion_charge) + 1
        else:
            n_ions = len(deprot_o)

        print(f'Polymer charge is {polymer_charge}')

        n_ions += extra_inserted # add the extra inserted ions to total added ions count
        counterion_charge = n_ions*ion_charge # calculate charge from added ions
        excess_charge = counterion_charge + polymer_charge # calculate excess charge from polymer and added ions

        print(f'Adding {n_ions} cations, which results in {excess_charge} excess charge in the system')

        # select which O's to place ions near
        print('WARNING: currently the only insertion algorithm is R-COOH groups with at least 2 water within 5 Angstroms')

        prot_idx = []
        for i,O in enumerate(prot_o):
            my_waters = self.universe.select_atoms(f'(type OW) and (sphzone 5 index {O.index})')
            if len(my_waters) > 1:
                prot_idx.append(i)

        if len(prot_idx) < extra_inserted:
            raise TypeError('Not enough R-COOH groups with waters')

        tmp = np.array(prot_o)
        o_group = mda.AtomGroup(deprot_o + tmp[prot_idx].tolist()) # select all deprotonated groups and extra_inserted random protonated groups

        # delete waters and find where to place ions
        ion_pos = []
        for O in o_group:
            my_id = O.index
            my_waters = self.universe.select_atoms(f'resname SOL and sphzone 5 index {my_id}').residues # select waters within 5 Angstroms
            if len(my_waters) > 0:
                pos = my_waters.atoms.center_of_mass() # place ion at the COM of the waters in the zone
                
                # randomly kick the ion around until it is at least 2 Angstroms away from all other atoms
                i = 0
                too_close = self.universe.select_atoms(f'point {pos[0]} {pos[1]} {pos[2]} {tol}') - my_waters.atoms
                n_close = len(too_close)
                while n_close > 0:
                    kick_vec = np.random.uniform(-1,1, size=3) # random direction for kick
                    kick_vec = kick_vec / np.linalg.norm(kick_vec) 
                    kick_strength = np.random.uniform(0,0.1) # kick is between 0 and 0.1 Angstroms

                    pos += kick_strength*kick_vec
                    too_close = self.universe.select_atoms(f'point {pos[0]} {pos[1]} {pos[2]} {tol}') - my_waters.atoms

                    if len(ion_pos) == 0: # if this is the first ion to be added, other ions do not matter
                        n_close = len(too_close)
                    else: # otherwise, make sure we do not place an ion on top of a previously inserted ion
                        dists = distances.distance_array(pos, np.array(ion_pos), box=self.universe.dimensions)
                        n_close = len(too_close) + (dists < 2).sum()
                    
                    i += 1
                    if i == 10_000:
                        print(f'failed on {O} with {n_close} atoms too close')
                        [print(f'\t{a}') for a in too_close]
                        exit()

                
                # print(f'Distance between O and ion: {distances.distance_array(pos,O.position, box=self.universe.dimensions)[0,0]:.4f}')
                print(f'Inserted ion near {O} after {i} kicks at position {pos}')
                ion_pos.append(pos) 
                self.atoms = self.atoms.subtract(my_waters.atoms) # remove the waters to be replaced

        # reassign residue numbers after deleted waters
        dims = self.universe.dimensions # save original dimensions
        n_residues = len(self.atoms.residues)
        resids = np.arange(1, n_residues+1)
        self.universe = mda.Merge(self.atoms)
        self.universe.add_TopologyAttr('resid', list(resids))

        # create an empty universe for the newly placed ions
        n_residues = len(self.atoms.residues)
        ion_u = mda.Universe.empty(n_ions,
                                   n_residues=n_ions,
                                   atom_resindex=list(range(n_ions)),
                                   trajectory=True)
        
        ion_u.add_TopologyAttr('name', [f'{ion_name}']*n_ions)
        ion_u.add_TopologyAttr('type', [f'{ion_name}']*n_ions)
        ion_u.add_TopologyAttr('resname', [f'{ion_name.upper()}']*n_ions)
        ion_u.add_TopologyAttr('resid', list(np.arange(n_residues+1, n_residues+n_ions+1)))

        ion_u.atoms.positions = np.array(ion_pos[:n_ions])

        # merge with system universe, write to gro file
        new_universe = mda.Merge(self.universe.atoms, ion_u.atoms)
        new_universe.dimensions = dims
        new_universe.atoms.write(output)

        return n_ions, output, excess_charge


    def random_insertion_in_membrane(self, PA_lims, ion_name='Na', ion_charge=1, extra_inserted=0, output='PA_ions.gro'):
        '''Add ions randomly to the membrane by merging universe with ion universe'''

        # locate COOH/COO- oxygens
        c_group = self.universe.select_atoms('type c and not bonded type n')
        deprot_o = []
        prot_o = []
        for c in c_group:
            my_Os = [atom for atom in c.bonded_atoms if atom.type == 'o']
            if len(my_Os) == 2:
                deprot_o.append(my_Os[0]) # add one of the two O's to the AtomGroup
            elif len(my_Os) == 1:
                prot_o.append(my_Os[0]) # add the =O to the AtomGroup
            else:
                raise TypeError(f'{c} is not the carbon in R-COOH or R-COO-')
        
        # calculate excess charge from adding ions
        polymer_charge = len(deprot_o)*-1
        if len(deprot_o) % ion_charge > 0: # if we cannot exactly balance polymer charge with cations
            n_ions = round(len(deprot_o) / ion_charge) + 1
        else:
            n_ions = len(deprot_o)

        print(f'Polymer charge is {polymer_charge}')

        n_ions += extra_inserted # add the extra inserted ions to total added ions count
        counterion_charge = n_ions*ion_charge # calculate charge from added ions
        excess_charge = counterion_charge + polymer_charge # calculate excess charge from polymer and added ions

        print(f'Adding {n_ions} cations, which results in {excess_charge} excess charge in the system')

        # find waters to replace with ions in the membrane (lower bound of PA + 5 to upper bound of PA - 5 to give a small buffer)
        waters = self.universe.select_atoms(f'(type {self.ow_type}) and (prop z >= {PA_lims[0]+5} and prop z <= {PA_lims[1]-5})')
        to_replace = self.atoms[np.random.choice(waters.indices, n_ions, replace=False)].residues
        ion_pos = np.zeros((n_ions, 3))
        for i in range(n_ions):
            ion_pos[i] = to_replace[i].atoms.center_of_mass()

        self.atoms = self.atoms.subtract(to_replace.atoms) # remove the waters to be replaced

        # reassign residue numbers after deleted waters
        dims = self.universe.dimensions # save original dimensions
        n_residues = len(self.atoms.residues)
        resids = np.arange(1, n_residues+1)
        self.universe = mda.Merge(self.atoms)
        self.universe.add_TopologyAttr('resid', list(resids))

        # create an empty universe for the newly placed ions
        n_residues = len(self.atoms.residues)
        ion_u = mda.Universe.empty(n_ions,
                                   n_residues=n_ions,
                                   atom_resindex=list(range(n_ions)),
                                   trajectory=True)
        
        ion_u.add_TopologyAttr('name', [f'{ion_name}']*n_ions)
        ion_u.add_TopologyAttr('type', [f'{ion_name}']*n_ions)
        ion_u.add_TopologyAttr('resname', [f'{ion_name.upper()}']*n_ions)
        ion_u.add_TopologyAttr('resid', list(np.arange(n_residues+1, n_residues+n_ions+1)))

        ion_u.atoms.positions = np.array(ion_pos[:n_ions])

        # merge with system universe, write to gro file
        new_universe = mda.Merge(self.universe.atoms, ion_u.atoms)
        new_universe.dimensions = dims
        new_universe.atoms.write(output)

        return n_ions, output, excess_charge
    

    def pair_insertion_in_membrane(self, cation_name='Na', anion_name='Cl', cation_charge=1, extra_inserted=0, tol=2, output='PA_ions.gro'):
        '''Add ions to the membrane by merging universe with ion universe'''

        from MDAnalysis.analysis import distances

        # locate COOH/COO- oxygens
        c_group = self.universe.select_atoms('type c and not bonded type n')
        deprot_o = []
        prot_o = []
        for c in c_group:
            my_Os = [atom for atom in c.bonded_atoms if atom.type == 'o']
            if len(my_Os) == 2:
                deprot_o.append(my_Os[0]) # add one of the two O's to the AtomGroup
            elif len(my_Os) == 1:
                prot_o.append(my_Os[0]) # add the =O to the AtomGroup
            else:
                raise TypeError(f'{c} is not the carbon in R-COOH or R-COO-')
        
        # calculate excess charge from adding ions
        polymer_charge = len(deprot_o)*-1
        if len(deprot_o) % cation_charge > 0: # if we cannot exactly balance polymer charge with cations
            n_ions = round(len(deprot_o) / cation_charge) + 1
        else:
            n_ions = len(deprot_o)

        print(f'Polymer charge is {polymer_charge}')

        n_ions += extra_inserted # add the extra inserted ions to total added ions count
        counterion_charge = n_ions*cation_charge # calculate charge from added ions
        excess_charge = polymer_charge # excess charge is the polymer charge since adding balanced ion pairs

        print(f'Adding {n_ions} ion pairs, which results in {excess_charge} excess charge in the system')

        # select which O's to place ions near
        print('WARNING: currently the only insertion algorithm is R-COOH groups with at least 2 water within 5 Angstroms')

        prot_idx = []
        for i,O in enumerate(prot_o):
            my_waters = self.universe.select_atoms(f'(type {self.ow_type}) and (sphzone 5 index {O.index})')
            if len(my_waters) > 1:
                prot_idx.append(i)

        if len(prot_idx) < extra_inserted:
            raise TypeError('Not enough R-COOH groups with waters')

        tmp = np.array(prot_o)
        o_group = mda.AtomGroup(deprot_o + tmp[prot_idx].tolist()) # select all deprotonated groups and extra_inserted random protonated groups

        # delete waters and find where to place ions
        cation_positions = []
        anion_positions = []
        to_replace = []
        for O in o_group:
            my_id = O.index
            my_waters = self.atoms.select_atoms(f'(type {self.ow_type}) and (sphzone 5 index {my_id})') # select waters within 5 Angstroms
            if len(my_waters) > 1:

                (idx1,idx2) = np.random.choice(my_waters.indices, 2, replace=False) # randomly select waters to replace
                to_replace.append(idx1)
                to_replace.append(idx2)
                ci_pos = self.atoms[idx1].position # place cation at the an OW for one of the waters
                ai_pos = self.atoms[idx2].position # place anion at the other water

                print(f'Attempting to place cation at {self.atoms[idx1]} and anion at {self.atoms[idx2]} by oxygen {O}')

                too_close_ci = self.universe.select_atoms(f'point {ci_pos[0]} {ci_pos[1]} {ci_pos[2]} {tol}') - self.atoms[idx1].residue.atoms - self.atoms[idx2].residue.atoms
                too_close_ai = self.universe.select_atoms(f'point {ai_pos[0]} {ai_pos[1]} {ai_pos[2]} {tol}') - self.atoms[idx1].residue.atoms - self.atoms[idx2].residue.atoms
                n_close = len(too_close_ai) + len(too_close_ci)

                ci_ai_dist = distances.distance_array(ci_pos, ai_pos, box=self.universe.dimensions)
                if ci_ai_dist < tol:
                    n_close += 1

                # kick the ions around until ions are within tolerance (i.e. not within tol Angstroms from other atoms)
                i = 0
                while n_close > 0:
                    if len(too_close_ci) > 0:
                        kick_vec = np.random.uniform(-1,1, size=3) # random direction for kick
                        kick_vec = kick_vec / np.linalg.norm(kick_vec) 
                        kick_strength = np.random.uniform(0,0.1) # kick is between 0 and 0.1 Angstroms
                        ci_pos += kick_strength*kick_vec

                    if len(too_close_ai) > 0:
                        kick_vec = np.random.uniform(-1,1, size=3) # random direction for kick
                        kick_vec = kick_vec / np.linalg.norm(kick_vec) 
                        kick_strength = np.random.uniform(0,0.1) # kick is between 0 and 0.1 Angstroms
                        ai_pos += kick_strength*kick_vec

                    too_close_ci = self.universe.select_atoms(f'point {ci_pos[0]} {ci_pos[1]} {ci_pos[2]} {tol}') - self.atoms[idx1].residue.atoms - self.atoms[idx2].residue.atoms
                    too_close_ai = self.universe.select_atoms(f'point {ai_pos[0]} {ai_pos[1]} {ai_pos[2]} {tol}') - self.atoms[idx1].residue.atoms - self.atoms[idx2].residue.atoms
                    n_close = len(too_close_ai) + len(too_close_ci)

                    # do not let cation and anion overlap either
                    ci_ai_dist = distances.distance_array(ci_pos, ai_pos, box=self.universe.dimensions)
                    if ci_ai_dist < tol:
                        n_close += 1

                    i += 1
                    if i == 10_000:
                        print(f'Failed on {O} with {n_close} atoms too close. Cation-anion distance is {ci_ai_dist[0,0]:.4f}')
                        [print(f'\t{a}') for a in too_close_ci+too_close_ai]
                        break # do not use this oxygen to add an ion pair

                if i != 10_000:
                    cation_positions.append(ci_pos)
                    anion_positions.append(ai_pos)
                    print(f'Placed ion pair after {i} iterations:')
                    print(f'\tDistance between O and cation: {distances.distance_array(ci_pos,O.position, box=self.universe.dimensions)[0,0]:.4f}')
                    print(f'\tDistance between O and anion: {distances.distance_array(ai_pos,O.position, box=self.universe.dimensions)[0,0]:.4f}')
                    print(f'\tDistance between cation and anion: {distances.distance_array(ci_pos,ai_pos, box=self.universe.dimensions)[0,0]:.4f}\n')

        self.atoms = self.atoms.subtract(self.atoms[np.array(to_replace)].residues.atoms) # remove the waters to be replaced
        
        # reassign residue numbers after deleted waters
        dims = self.universe.dimensions # save original dimensions
        n_residues = len(self.atoms.residues)
        resids = np.arange(1, n_residues+1)
        self.universe = mda.Merge(self.atoms)
        self.universe.add_TopologyAttr('resid', list(resids))

        # create an empty universe for the newly placed ions
        n_residues = len(self.atoms.residues)
        ion_u = mda.Universe.empty(n_ions*2,
                                n_residues=n_ions*2,
                                atom_resindex=list(range(n_ions*2)),
                                trajectory=True)

        ion_u.add_TopologyAttr('name', [f'{cation_name}']*n_ions+[f'{anion_name}']*n_ions)
        ion_u.add_TopologyAttr('type', [f'{cation_name}']*n_ions+[f'{anion_name}']*n_ions)
        ion_u.add_TopologyAttr('resname', [f'{cation_name.upper()}']*n_ions+[f'{anion_name.upper()}']*n_ions)
        ion_u.add_TopologyAttr('resid', list(np.arange(n_residues+1, n_residues+n_ions*2+1)))

        ion_u.atoms.positions = np.array(cation_positions[:n_ions] + anion_positions[:n_ions])

        # merge with system universe, write to gro file
        new_universe = mda.Merge(self.universe.atoms, ion_u.atoms)
        new_universe.dimensions = dims
        new_universe.atoms.write(output)

        return n_ions, output, excess_charge


    def random_amides(self, num, seed=12345):
        '''Randomly select num of amide bonds to break'''

        sel = self.universe.select_atoms(f'type {self.xlink_n_type}')
        n = len(sel)
        rng = np.random.default_rng(seed=seed)
        rand_idx = rng.choice(np.arange(0,n), size=num, replace=False)
        
        return sel[rand_idx]


    def remove_water_for_Cl(self, n, radius_for_water_search=6, n_waters=4, tol=2):
        my_waters = self.universe.atoms.select_atoms(f'(type {self.ow_type}) and (sphzone {radius_for_water_search} index {n.index})')
        if len(my_waters) < n_waters:
            print('Not enough waters found near amide N atom index', n.index)
            return None
        
        # # save the positions of the waters to be replaced
        # original_pos = my_waters[0].position.copy()
        # pos = my_waters[0].position

        # estimate where Cl should be based on n, hn
        hn = [a for a in n.bonded_atoms if a.type == 'hn'][0]
        d = 1.763 # hard-code force field parameters for now
        original_pos = solve_for_Cl_position(hn.position, n.position, d)
        pos = original_pos.copy()

        # remove the water and rebuild a universe with fewer residues
        remaining_atoms = self.universe.atoms - my_waters[:n_waters].residues.atoms
        new_universe = mda.Merge(remaining_atoms)
        new_universe.dimensions = self.universe.dimensions
        self.universe = new_universe
        self.atoms = self.universe.atoms

        # reassign residue numbers after deleted waters
        n_residues = len(self.universe.atoms.residues)
        resids = np.arange(1, n_residues+1)
        self.universe.add_TopologyAttr('resid', list(resids))

        # # randomly kick the Cl around until it is at least 2 Angstroms away from all other atoms
        # i = 0
        # too_close = self.universe.select_atoms(f'point {pos[0]} {pos[1]} {pos[2]} {tol}')
        # n_close = len(too_close)
        # while n_close > 0:
        #     kick_vec = np.random.uniform(-1,1, size=3) # random direction for kick
        #     kick_vec = kick_vec / np.linalg.norm(kick_vec) 
        #     kick_strength = np.random.uniform(0,0.1) # kick is between 0 and 0.1 Angstroms

        #     pos += kick_strength*kick_vec
        #     too_close = self.universe.select_atoms(f'point {pos[0]} {pos[1]} {pos[2]} {tol}')

        #     # check distance to the N it will be bonded to
        #     dists = distances.distance_array(pos, n.position, box=self.universe.dimensions)
        #     print(f'\n\tDistance between Cl and N: {dists[0,0]:.4f}')
        #     n_close = len(too_close)
        #     print(f'\t{n_close} atoms are too close to Cl: {[a for a in too_close]}')
            
        #     i += 1
        #     if i == 50_000:
        #         print(f'failed on {n} with {n_close} atoms too close')
        #         [print(f'\t{a}') for a in too_close]
        #         exit()

        #     if dists[0,0] > 6: # start over if Cl is too far from N
        #         pos = original_pos.copy()
        #         i = 0
        #         print(f'\tReset position to {pos}')

        return pos


    def remove_water_for_OH(self, c, Cl_position, radius_for_water_search=6, n_waters=4, protonate=False, tol=2):
        '''Remove water to make room for OH group'''
        my_waters = self.universe.atoms.select_atoms(f'(type {self.ow_type}) and (sphzone {radius_for_water_search} index {c.index})')
        if len(my_waters) < n_waters:
            print('\tNot enough waters found near amide C atom index', c.index)
            if protonate:
                return None, None
            else:
                return None

        # estimate where O should be based on c, o, ca
        o = [a for a in c.bonded_atoms if a.type == 'o'][0]
        ca = [a for a in c.bonded_atoms if a.type == 'ca'][0]
        theta = 122.88 # hard-code force field parameters for now
        d = 1.306 # c-oh bond length in Angstroms
        original_O_pos,original_H_pos = solve_for_OH_position(o.position, c.position, ca.position, theta, d)
        O_position = original_O_pos.copy()
        H_position = original_H_pos.copy()

        # place the O at the N, since it is closest to where it needs to be
        # n = [a for a in c.bonded_atoms if a.type == self.xlink_n_type][0]
        # original_O_pos = n.position.copy()
        # original_H_pos = n.position.copy() + np.array([0.0, 0.0, 0.96]) # place H 0.96 Angstroms away from O initially
        # O_position = n.position
        # H_position = n.position + np.array([0.0, 0.0, 0.96])

        # print('\tOriginal O position:', original_O_pos)
        # print('\tOriginal H position:', original_H_pos)

        # # save the positions of the waters to be replaced
        # original_O_pos = my_waters[0].position.copy()
        # original_H_pos = my_waters[0].bonded_atoms[1].position.copy()
        # O_position = my_waters[0].position
        # H_position = my_waters[0].bonded_atoms[1].position  # for 50% deprotonated, we want to add O- for half and OH for half

        # # randomly kick the O around until it is at least 2 Angstroms away from all other atoms
        # i = 0
        # too_close = self.universe.select_atoms(f'point {O_position[0]} {O_position[1]} {O_position[2]} {tol}') - my_waters[:n_waters].residues.atoms
        # n_close = len(too_close)
        # while n_close > 0:
        #     kick_vec = np.random.uniform(-1,1, size=3) # random direction for kick
        #     kick_vec = kick_vec / np.linalg.norm(kick_vec) 
        #     kick_strength = np.random.uniform(0,0.1) # kick is between 0 and 0.1 Angstroms

        #     O_position += kick_strength*kick_vec
        #     H_position += kick_strength*kick_vec # drag H along with O
        #     too_close_dists = distances.distance_array(O_position, (self.universe.atoms - my_waters[:n_waters].residues.atoms).positions, box=self.universe.dimensions)
        #     print(f'\n\tDistances between O and other atoms below tolerance: {too_close_dists[too_close_dists < tol]}')
        #     n_close = (too_close_dists < tol).sum()
        #     too_close = self.universe.select_atoms(f'point {O_position[0]} {O_position[1]} {O_position[2]} {tol}') - my_waters[:n_waters].residues.atoms
        #     print(f'\tToo close atoms: {[a for a in too_close]}')

        #     # make sure it does not overlap the inserted Cl
        #     dists = distances.distance_array(O_position, Cl_position, box=self.universe.dimensions)
        #     print(f'\tDistance between O and Cl: {dists[0,0]:.4f}')
        #     n_close += (dists < tol).sum()

        #     # check distance to the C it will be bonded to
        #     dists = distances.distance_array(O_position, c.position, box=self.universe.dimensions)
        #     print(f'\tDistance between O and C: {dists[0,0]:.4f}')

        #     i += 1
        #     if i == 50_000:
        #         print(f'\tfailed on {c} with {n_close} atoms too close')
        #         [print(f'\t{a}') for a in too_close]
        #         exit()

        #     if dists[0,0] > 6: # start over if O is too far from C
        #         O_position = original_O_pos.copy()
        #         H_position = original_H_pos.copy()
        #         i = 0
        #         print(f'\tReset position to {O_position}')


        # remove the water and rebuild a universe with fewer residues
        print(f'\tDeleting water atoms {my_waters[:n_waters].residues.atoms}')
        remaining_atoms = self.universe.atoms - my_waters[:n_waters].residues.atoms
        new_universe = mda.Merge(remaining_atoms)
        new_universe.dimensions = self.universe.dimensions
        self.universe = new_universe
        self.atoms = self.universe.atoms

        # reassign residue numbers after deleted waters
        n_residues = len(self.universe.atoms.residues)
        resids = np.arange(1, n_residues+1)
        self.universe.add_TopologyAttr('resid', list(resids))

        if protonate:
            return O_position, H_position
        else:
            return O_position


    def reassign_residue_indices(self):
        '''Reassign residue indices after breaking bonds and creating new molecules'''

        # Reassign residue indices based on connected components
        atom_to_new_resindex = {}
        new_resindex_to_atom = {}
        n_new_residues = 0

        for residue in self.polymer.residues:
            res_atoms = residue.atoms
            res_graph = build_graph_from_atoms(res_atoms)
            
            # Get connected components for this residue
            components = list(nx.connected_components(res_graph))
            
            for component in components:
                new_resindex_to_atom[n_new_residues] = []
                # Assign all atoms in this component to the same new residue
                for atom_idx in component:
                    atom_to_new_resindex[atom_idx] = n_new_residues
                    new_resindex_to_atom[n_new_residues].append(atom_idx)
                
                new_resindex_to_atom[n_new_residues] = np.array(new_resindex_to_atom[n_new_residues])
                n_new_residues += 1

        # Create a new empty universe with correct number of residues
        n_atoms = len(self.polymer.atoms)
        new_polymer_u = mda.Universe.empty(
            n_atoms=n_atoms,
            n_residues=n_new_residues,
            atom_resindex=[atom_to_new_resindex[atom.index] for atom in self.polymer.atoms],
            trajectory=True
        )

        # Copy over topology attributes
        new_polymer_u.add_TopologyAttr('name', [atom.name for atom in self.polymer.atoms])
        new_polymer_u.add_TopologyAttr('type', [atom.type for atom in self.polymer.atoms])
        new_polymer_u.add_TopologyAttr('element', [atom.element for atom in self.polymer.atoms])
        new_polymer_u.add_TopologyAttr('mass', [atom.mass for atom in self.polymer.atoms])
        new_polymer_u.add_TopologyAttr('resname', [f'PA{i}' for i in range(1, n_new_residues + 1)])
        new_polymer_u.add_TopologyAttr('resid', [i for i in range(1, n_new_residues + 1)])

        # Copy positions and bonds
        new_polymer_u.atoms.positions = self.polymer.atoms.positions
        bond_indices = [(b.atoms[0].index, b.atoms[1].index) for b in self.polymer.bonds]
        new_polymer_u.add_TopologyAttr('bonds', bond_indices)

        # Merge with non-polymer atoms
        waters = self.universe.select_atoms(f'type {self.ow_type} {self.hw_type}')
        ions = self.universe.select_atoms(f'type {self.cation_type} {self.anion_type}')
        merged_u = mda.Merge(new_polymer_u.atoms, waters+ions)
        merged_u.dimensions = self.universe.dimensions

        # Reassign resids to be sequential
        # Polymer residues: 1 to n_new_residues
        # Non-polymer residues: n_new_residues+1 onwards
        non_polymer_start_idx = n_new_residues + 1

        # Get the non-polymer residues and renumber them
        for i, res in enumerate(merged_u.residues[n_new_residues:], start=non_polymer_start_idx):
            for atom in res.atoms:
                atom.residue.resid = i

        # Replace the universe
        self.universe = merged_u
        print(f'\tNumber of new polymer residues: {n_new_residues}')

        return atom_to_new_resindex, new_resindex_to_atom


    def update_topology_after_breaking_bond(self, top, n, c, new_resindex_to_atom, atom_to_new_resindex):
        '''Update topology file after breaking amide bonds'''

        # update the topology file
        # my GromacsTopology class writes out only using the molecules, so it should be fine if we update all the molecule information
        from gromacs_topology import Molecule

        polymer_idx = self.polymer.indices
        broken_amide_N_idx = n.index
        broken_amide_C_idx = c.index
        n_new_residues = len(new_resindex_to_atom)

        # initialize new molecules
        new_molecules = []
        for i in range(1, n_new_residues + 1):
            mol = Molecule(f'PA{i}', 1)
            setattr(mol, 'atoms', [])
            setattr(mol, 'bonds', [])
            setattr(mol, 'angles', [])
            setattr(mol, 'dihedrals', [])

            mol.directives = ['bonds', 'angles', 'dihedrals']
            new_molecules.append(mol)

        # loop through all atoms and assign to new molecules
        for atom in top.atoms[polymer_idx]:
            # add to new molecule
            resindex = atom_to_new_resindex[atom.idx]
            my_map = {orig_idx : new_idx+1 for new_idx,orig_idx in enumerate(new_resindex_to_atom[resindex])} # renumbering map within new residue
            
            # update atom attributes for new molecule
            atom.num = my_map[atom.idx]
            atom.resnum = resindex + 1
            atom.resname = f'PA{resindex + 1}'

            my_mol = new_molecules[resindex]
            my_mol.atoms.append(atom)

        # sort the atoms in each new molecule by their new atom number
        for mol in new_molecules:
            mol.atoms.sort(key=lambda a: a.num)

        # loop through all bonds and assign to new molecules
        n_bonds_skipped = 0
        for bond in top.bonds: # this is all polymer bonds because water uses settles and monoatomic ions have no bonds
            atom1, atom2 = bond.atoms
            my_types = [a.type for a in bond.atoms]
            my_mol = new_molecules[atom_to_new_resindex[atom1.idx]]

            if 'c' in my_types and 'n' in my_types:
                c_atom = atom1 if atom1.type == 'c' else atom2
                n_atom = atom1 if atom1.type == 'n' else atom2
                if c_atom.idx == broken_amide_C_idx and n_atom.idx == broken_amide_N_idx:
                    # skip amide bonds that have been broken, which deletes them from the new molecule
                    # print(f'\t\tSkipping bond between atoms {atom1.idx}:{atom1.type} and {atom2.idx}:{atom2.type}')
                    n_bonds_skipped += 1
                    continue

            my_mol.bonds.append(bond)

        print(f'\t\tSkipped {n_bonds_skipped} bonds corresponding to broken amide bonds')

        # loop through all angles and assign to new molecules
        n_angles_skipped = 0
        n_angles_added = 0
        for angle in top.angles: # this is all polymer angles because water uses settles and monoatomic ions have no angles
            atom1, atom2, atom3 = angle.atoms
            my_types = [a.type for a in angle.atoms]
            my_mol = new_molecules[atom_to_new_resindex[atom1.idx]]

            if 'c' in my_types and 'n' in my_types:
                c_atom = [atom for atom in angle.atoms if atom.type == 'c'][0]
                n_atom = [atom for atom in angle.atoms if atom.type == 'n'][0]
                if c_atom.idx == broken_amide_C_idx and n_atom.idx == broken_amide_N_idx:
                    # skip angles that involve broken amide bonds
                    # print(f'\t\tSkipping angle between atoms {atom1.idx}:{atom1.type} and {atom2.idx}:{atom2.type} and {atom3.idx}:{atom3.type}')
                    n_angles_skipped += 1
                    continue

            my_mol.angles.append(angle)
            n_angles_added += 1

        print(f'\t\tSkipped {n_angles_skipped} angles corresponding to broken amide bonds')

        # loop through all dihedrals and assign to new molecules
        n_dihedrals_skipped = 0
        for dihedral in top.dihedrals: # this is all polymer dihedrals because water uses settles and monoatomic ions have no dihedrals
            atom1, atom2, atom3, atom4 = dihedral.atoms
            my_types = [a.type for a in dihedral.atoms]
            my_mol = new_molecules[atom_to_new_resindex[atom1.idx]]

            if 'c' in my_types and 'n' in my_types:
                c_atom = [atom for atom in dihedral.atoms if atom.type == 'c'][0]
                n_atom = [atom for atom in dihedral.atoms if atom.type == 'n'][0]
                if c_atom.idx == broken_amide_C_idx and n_atom.idx == broken_amide_N_idx:
                    # skip dihedrals that involve broken amide bonds
                    # print(f'\t\tSkipping dihedral between atoms {atom1.idx}:{atom1.type} and {atom2.idx}:{atom2.type} and {atom3.idx}:{atom3.type} and {atom4.idx}:{atom4.type}')
                    n_dihedrals_skipped += 1
                    continue

            my_mol.dihedrals.append(dihedral)

        print(f'\t\tSkipped {n_dihedrals_skipped} dihedrals corresponding to broken amide bonds')

        # add the water and ion molecules back
        for mol in top.molecules:
            if not mol.name.startswith('PA'):
                new_molecules.append(mol)

        # update topology molecules
        top.molecules = new_molecules
        
        return top
        

    def get_global_atom_indices(self, universe, top, breaking_bonds=False, n_counterions=0):
        '''Get global atom indices from universe'''

        # reassign global atom indices to the GRO file
        global_map = {}
        new_idx = 0

        # start with polymer atoms

        if breaking_bonds: # use the top indices when breaking bonds but not when adding atoms
            for mol in top.molecules[:-4]:  # last 4 molecules are non-polymer
                for atom in mol.atoms:
                    global_map[atom.idx] = new_idx
                    new_idx += 1
        else:
            n_polymer_mols = len(top.molecules) - 4  # last 4 molecules are non-polymer
            for i in range(n_polymer_mols):
                polymer_res = universe.select_atoms(f'resname PA{i+1}')
                for atom in polymer_res:
                    global_map[atom.ix] = new_idx
                    new_idx += 1

        # then non-polymer atoms (first, waters)
        for atom in universe.select_atoms('resname SOL'):
            global_map[atom.ix] = new_idx
            new_idx += 1

        # now for the number of counterions added to balance deprotonated O's
        print('\n\tCounterions added: ', n_counterions)
        NA_idx = 0

        if n_counterions > 0:
            for n in range(1, n_counterions+1):
                n_mols_NA = top.molecules[-3-n].n_mols
                print(f'\t\tAdding {n_mols_NA} Na ions from topology molecule index {-3-n}')
                for atom in universe.select_atoms(f'resname NA')[NA_idx:NA_idx+n_mols_NA]:
                    print(f'\t\tatom.ix {atom.ix} --> new_idx {new_idx}')
                    global_map[atom.ix] = new_idx
                    new_idx += 1
                NA_idx += n_mols_NA

        print('\t\tNew NA idx: ', NA_idx)
        n_mols_NA_1 = top.molecules[-3].n_mols # NA molecules are split in topology, this is the set from original genions
        for atom in universe.select_atoms('resname NA')[NA_idx:n_mols_NA_1+NA_idx]:
            global_map[atom.ix] = new_idx
            new_idx += 1

        # then Cl ions
        for atom in universe.select_atoms('resname CL'):
            global_map[atom.ix] = new_idx
            new_idx += 1

        # then all other Na ions
        for atom in universe.select_atoms('resname NA')[n_mols_NA_1+NA_idx:]:
            global_map[atom.ix] = new_idx
            new_idx += 1

        return global_map


    def add_new_atoms_for_chlorination(self, n, c, Cl_position, O_position, H_position=None):
        '''Add new Cl and OH atoms to the universe after breaking amide bonds'''

        # create an empty universe for the newly placed atoms
        if H_position is not None:
            new_atom_resindices = np.array([n.resindex, c.resindex, c.resindex]) # add to the same residues as the broken amide N and C
            n_added = 3
            positions = np.vstack([Cl_position, O_position, H_position])
            names = np.array(['Cl', 'O', 'H'])
        else:
            new_atom_resindices = np.array([n.resindex, c.resindex]) # add to the same residues as the broken amide N and C
            n_added = 2
            positions = np.vstack([Cl_position, O_position])
            names = np.array(['Cl', 'O'])

        new_atoms_u = mda.Universe.empty(
            n_atoms=n_added,
            n_residues=len(self.universe.residues),  # Don't create new residues, but use the existing residues
            atom_resindex=new_atom_resindices,
            trajectory=True
        )

        new_atoms_u.add_TopologyAttr('name', names)
        new_atoms_u.add_TopologyAttr('type', names) # does not matter that these are not correct atomtypes
        new_atoms_u.add_TopologyAttr('element', names)
        new_atoms_u.add_TopologyAttr('resname', [res.resname for res in self.universe.residues])
        new_atoms_u.add_TopologyAttr('resid', [res.resid for res in self.universe.residues])
        new_atoms_u.atoms.positions = positions

        # Merge with existing universe
        merged_u = mda.Merge(self.universe.atoms, new_atoms_u.atoms)
        merged_u.dimensions = self.universe.dimensions

        print(f'\tOriginal number of atoms: {len(self.universe.atoms)}')
        print(f'\tAdded {n_added} new atoms to existing residues')
        print(f'\tTotal atoms: {len(merged_u.atoms)}')

        return merged_u
    

    def update_topology_with_Cl(self, top, n, param_gro, param_top):
        '''Get the bonded parameters and add them to the topology file'''

        from gromacs_topology import Atom, Bond, Angle, Dihedral

        # get representative atoms from parameterized fragment
        cl = param_gro.universe.select_atoms('type cl')
        cl_atom = param_top.atoms[cl.indices[0]]
        nh_type = param_top.atomtypes['nh'] # broken amide N changes to type nh (amine N)

        # add Cl to the N on broken amide bonds
        total_atoms = len(self.universe.atoms)
        n_top = top.atoms[n.index]

        # pull most attributes from the parameterized fragment with Cl
        atype = cl_atom.type
        atomname = cl_atom.atomname
        cgnr = cl_atom.cgnr
        charge = cl_atom.charge
        mass = cl_atom.mass
        atomtype = cl_atom.atomtype

        # pull other attributes from the current molecule
        num = n_top.molecule.n_atoms + 1
        resnum = n_top.resnum
        resname = n_top.resname
        molecule = n_top.molecule

        # create new atom and add to molecule
        new_atom_line = f'{num:7d} {atype:4s}\t\t{resnum} {resname:8s} {atomname:9s}\t{cgnr:d}\t{charge:.8f}\t{mass:.8f}\n'
        new_atom = Atom(total_atoms, new_atom_line)
        new_atom.atomtype = atomtype
        new_atom.molecule = molecule
        molecule.atoms.append(new_atom)

        # change N atom type from 'n' to 'nh'
        n_top.atomtype = nh_type
        n_top.type = 'nh'

        # add new bonds, angles, dihedrals to the molecule
        nh_cl_bond = param_top.get_bond('nh', 'cl')
        my_nh_cl_bond = Bond(new_atom, n_top, nh_cl_bond.type, nh_cl_bond._params)
        molecule.bonds.append(my_nh_cl_bond)

        hn_on_N = [a for a in n.bonded_atoms if a.type == 'hn'][0]
        hn_top = top.atoms[hn_on_N.index]
        hn_nh_cl_angle = param_top.get_angle('hn', 'nh', 'cl')
        my_hn_nh_cl_angle = Angle(hn_top, n_top, new_atom, hn_nh_cl_angle.type, hn_nh_cl_angle._params)
        molecule.angles.append(my_hn_nh_cl_angle)

        ca_on_N = [a for a in n.bonded_atoms if a.type == 'ca'][0]
        ca_top = top.atoms[ca_on_N.index]
        ca_nh_cl_angle = param_top.get_angle('ca', 'nh', 'cl')
        my_ca_nh_cl_angle = Angle(ca_top, n_top, new_atom, ca_nh_cl_angle.type, ca_nh_cl_angle._params)
        molecule.angles.append(my_ca_nh_cl_angle)

        for ang in n_top.angles:
            a1, a2, a3 = ang.atoms
            my_types = [a.type for a in ang.atoms]

            if Counter(my_types) == Counter(['ca', 'nh', 'hn']):
                continue  # this improper is skipped in the moltemplate GAFF topology
            
            dih = param_top.get_dihedral(a1.type, a2.type, a3.type, new_atom.type) 
            my_dihedral = Dihedral(a1, a2, a3, new_atom, dih.type, dih._params) # WARNING: make sure these are in the correct order
            molecule.dihedrals.append(my_dihedral) # middle two atoms need to be the middle atoms of the dihedral

        print(f'\tAdded Cl atom to broken amide N atom index {n.index} as atom index {total_atoms}')
        total_atoms += 1

        return top, total_atoms
    

    def update_topology_with_O(self, top, c, total_atoms, param_gro, param_top, protonate=False):
        '''Get the bonded parameters for the new O atom after breaking crosslinks and add them to the topology file'''

        from gromacs_topology import Atom, Bond, Angle, Dihedral

        term_c = param_gro.universe.select_atoms('(type c) and (not bonded type n)')
        o = param_gro.universe.select_atoms('(type o) and (bonded group term_c)', term_c=term_c)
        o_atom = param_top.atoms[o.indices[0]]
        oh = param_gro.universe.select_atoms('type oh')
        oh_atom = param_top.atoms[oh.indices[0]]

        # add O to the C on broken amide bonds
        my_C_top = top.atoms[c.index]

        # add O atom
        if protonate:
            ref_atom = oh_atom
        else:
            ref_atom = o_atom

        # pull most attributes from the parameterized fragment
        atype = ref_atom.type
        atomname = ref_atom.atomname
        cgnr = ref_atom.cgnr
        charge = ref_atom.charge
        mass = ref_atom.mass
        atomtype = ref_atom.atomtype

        # pull other attributes from the current molecule
        num = my_C_top.molecule.n_atoms + 1
        resnum = my_C_top.resnum
        resname = my_C_top.resname
        molecule = my_C_top.molecule

        # create new atom and add to molecule
        new_atom_line = f'{num:7d} {atype:4s}\t\t{resnum} {resname:8s} {atomname:9s}\t{cgnr:d}\t{charge:.8f}\t{mass:.8f}\n'
        new_atom = Atom(total_atoms, new_atom_line)
        new_atom.atomtype = atomtype
        new_atom.molecule = molecule
        molecule.atoms.append(new_atom)

        # add new bonds, angles, dihedrals to the molecule
        b = param_top.get_bond('c', new_atom.type)
        my_o_c_bond = Bond(new_atom, my_C_top, b.type, b._params)
        molecule.bonds.append(my_o_c_bond)

        my_C_top_bonded_atoms = []
        new_angles = []
        for b in my_C_top.bonds:

            my_C_top_bonded_atoms.append([a for a in b.atoms if a != my_C_top][0])

            # need my_C_top to be in the middle of the angle
            # new o atom will always be an "outside" atom bonded to the c
            # for example, o-c-o
            a1, a2 = b.atoms
            other_atom = a1 if a2 == my_C_top else a2

            angle = param_top.get_angle(a1.type, a2.type, new_atom.type)
            my_angle = Angle(other_atom, my_C_top, new_atom, angle.type, angle._params)
            molecule.angles.append(my_angle)
            new_angles.append(my_angle)

        new_dihedrals = []
        for ang in my_C_top.angles:
            a1, a2, a3 = ang.atoms
            my_types = [a.type for a in ang.atoms]

            if Counter(my_types) == Counter(['c', 'o', 'ca']):
                is_improper = True # this is an improper
            else:
                is_improper = False

            dih = param_top.get_dihedral(a1.type, a2.type, a3.type, new_atom.type, is_improper=is_improper)
            
            # for impropers, the 3rd atom in the list is the central atom
            # for example, ca-o-c-oh is an improper for a carboxylic acid on benzene ring
            # for proper dihedrals, the middle two atoms are the middle atoms of the dihedral
            # and the outer two atoms are bonded to the nearest middle atom
            # for example, o-c-oh-ho is a proper dihedral for a carboxylic acid

            others = [at for at in ang.atoms if at is not my_C_top]
            other_atom2 = [o for o in others if o in my_C_top_bonded_atoms][0] # this one must be bonded to my_C_top
            other_atom1 = [o for o in others if o is not other_atom2][0]

            if is_improper:
                # as long as my_C_top is the 3rd atom, the order of the other atoms does not matter
                my_dihedral = Dihedral(other_atom1, other_atom2, my_C_top, new_atom, dih.type, dih._params)
            else:
                # if we set my_C_top as the 3rd atom, the new_atom has to be the 4th atom
                my_dihedral = Dihedral(other_atom1, other_atom2, my_C_top, new_atom, dih.type, dih._params)
            
            molecule.dihedrals.append(my_dihedral)
            new_dihedrals.append(my_dihedral)

        new_atom.bonds.append(my_o_c_bond)
        new_atom.angles.extend(new_angles)
        new_atom.dihedrals.extend(new_dihedrals)

        # add them to C atom as well
        my_C_top.bonds.append(my_o_c_bond)
        my_C_top.angles.extend(new_angles)
        my_C_top.dihedrals.extend(new_dihedrals)

        print(f'\tAdded O atom to broken amide C atom index {c.index} as atom index {total_atoms}')
        total_atoms += 1

        return top, total_atoms


    def update_topology_with_H(self, top, c, total_atoms, param_gro, param_top):
        '''Get the bonded parameters for the new H atom after breaking crosslinks (only for protonated OH) and add them to the topology file'''

        from gromacs_topology import Atom, Bond, Angle, Dihedral

        ho = param_gro.universe.select_atoms('type ho')
        ho_atom = param_top.atoms[ho.indices[0]]

        # add H to the C on broken amide bonds
        my_C_top = top.atoms[c.index]
    
        for b in my_C_top.bonds:
            has_oh = len([a for a in b.atoms if a.type == 'oh']) == 1
            if has_oh:
                my_O_top = [a for a in b.atoms if a.type == 'oh'][0]
        
        # pull most attributes from the parameterized fragment
        ref_atom = ho_atom
        atype = ref_atom.type
        atomname = ref_atom.atomname
        cgnr = ref_atom.cgnr
        charge = ref_atom.charge
        mass = ref_atom.mass
        atomtype = ref_atom.atomtype

        # pull other attributes from the current molecule
        num = my_C_top.molecule.n_atoms + 1
        resnum = my_C_top.resnum
        resname = my_C_top.resname
        molecule = my_C_top.molecule

        # create new atom and add to molecule
        new_H_atom_line = f'{num:7d} {atype:4s}\t\t{resnum} {resname:8s} {atomname:9s}\t{cgnr:d}\t{charge:.8f}\t{mass:.8f}\n'
        new_H_atom = Atom(total_atoms, new_H_atom_line)
        new_H_atom.atomtype = atomtype
        new_H_atom.molecule = molecule
        molecule.atoms.append(new_H_atom)

        # add new bonds, angles, dihedrals to the molecule
        b = param_top.get_bond('oh', new_H_atom.type)
        my_oh_ho_bond = Bond(my_O_top, new_H_atom, b.type, b._params)
        molecule.bonds.append(my_oh_ho_bond)

        my_O_top_bonded_atoms = []
        for b in my_O_top.bonds:
            a1, a2 = b.atoms
            my_O_top_bonded_atoms.append([a for a in b.atoms if a != my_O_top][0])

            other_atom = a1 if a2 == my_O_top else a2
            angle = param_top.get_angle(a1.type, a2.type, new_H_atom.type)
            my_angle = Angle(other_atom, my_O_top, new_H_atom, angle.type, angle._params) # the hydrogen should always be 3rd atom
            molecule.angles.append(my_angle)

        for ang in my_O_top.angles:
            a1, a2, a3 = ang.atoms

            # this should be set up properly, but let's be sure that my_O_top is the 3rd atom and new_H_atom is last
            others = [at for at in ang.atoms if at is not my_O_top]
            other_atom2 = [o for o in others if o in my_O_top_bonded_atoms][0] # this one must be bonded to my_O_top
            other_atom1 = [o for o in others if o is not other_atom2][0]
            
            dih = param_top.get_dihedral(a1.type, a2.type, a3.type, new_H_atom.type)
            my_dihedral = Dihedral(other_atom1, other_atom2, my_O_top, new_H_atom, dih.type, dih._params)
            molecule.dihedrals.append(my_dihedral)
            # print(f'\tAdded dihedral: {other_atom1.idx}-{other_atom2.idx}-{my_O_top.idx}-{new_H_atom.idx} with types {other_atom1.type}-{other_atom2.type}-{my_O_top.type}-{new_H_atom.type}')

        print(f'\tAdded H atom to broken amide C atom index {c.index} as atom index {total_atoms}')
        total_atoms += 1

        print('\n\tTotal atoms in topology:', total_atoms)

        return top, total_atoms


    def reassign_charges_after_chlorination(self, top):
        '''Reassign charges on atoms after chlorination by breaking polymer into its fragments'''

        from monomer_classes import MPD, MPD_Cl_T, TMC

        atoms = self.polymer

        original_charges = atoms.charges

        with open('charges_modified.yaml', 'r') as file: # read in charges from yaml file
            charges_dict = yaml.safe_load(file)

        print('\tCreating MPD monomer fragments...')
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

        print('\tCreating TMC monomer fragments...')
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
        print('\n\tTotal charge before reassignment: {:.4f}'.format(original_charges.sum()))
        print('\tTotal charge after reassignment: {:.4f}'.format(new_charges.sum()))
        
        return top


    def calculate_charges_after_chlorination(self):
        '''Calculate partial charges after chlorination by breaking polymer into its fragments'''

        from monomer_classes import MPD, MPD_Cl_T, TMC

        # read in original charges
        with open('charges.yaml', 'r') as file:
            charges_dict = yaml.safe_load(file)

        atoms = self.universe.select_atoms(f'not type {self.ow_type} {self.hw_type} {self.anion_type} {self.cation_type}') # polymer atoms only

        # get the number of each fragment
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


        n_mpd_mono = len([mono for mono in mpds if mono.name == 'MPD'])
        n_mpd_mono_Cl = len([mono for mono in mpds if mono.name in ['MPD_Cl_1', 'MPD_Cl_2']])
        n_mpd_mono_2Cl = len([mono for mono in mpds if mono.name == 'MPD_2Cl'])
        n_mpd_L = len([mono for mono in mpds if mono.name == 'MPD_L'])
        n_mpd_T = len([mono for mono in mpds if mono.name == 'MPD_T'])
        n_mpd_T_Cl = len([mono for mono in mpds if mono.name in ['MPD_T_Cl_1', 'MPD_T_Cl_2']])
        n_tmc_mono = len([mono for mono in tmcs if mono.name == 'TMC'])
        n_tmc_mono_0P = len([mono for mono in tmcs if mono.name == 'TMC_0P'])
        n_tmc_mono_1P = len([mono for mono in tmcs if mono.name in ['TMC_1P_1', 'TMC_1P_2', 'TMC_1P_3']])
        n_tmc_mono_2P = len([mono for mono in tmcs if mono.name in ['TMC_2P_1', 'TMC_2P_2', 'TMC_2P_3']])
        n_tmc_mono_3P = len([mono for mono in tmcs if mono.name == 'TMC_3P'])
        n_tmc_LD = len([mono for mono in tmcs if mono.name in ['TMC_L_D_1', 'TMC_L_D_2', 'TMC_L_D_3']])
        n_tmc_LP = len([mono for mono in tmcs if mono.name in ['TMC_L_P_1', 'TMC_L_P_2', 'TMC_L_P_3']])
        n_tmc_T0P = len([mono for mono in tmcs if mono.name in ['TMC_T_0P_1', 'TMC_T_0P_2', 'TMC_T_0P_3']])
        n_tmc_T1P = len([mono for mono in tmcs if mono.name in ['TMC_T_1P_1_2', 'TMC_T_1P_1_3', 'TMC_T_1P_2_1', 'TMC_T_1P_2_3', 'TMC_T_1P_3_1', 'TMC_T_1P_3_2']])
        n_tmc_T2P = len([mono for mono in tmcs if mono.name in ['TMC_T_2P_1', 'TMC_T_2P_2', 'TMC_T_2P_3']])
        n_tmc_C = len([mono for mono in tmcs if mono.name == 'TMC_C'])

        mpd_L = get_charge('MPD_L', charges_dict)
        mpd_T = get_charge('MPD_T', charges_dict)
        mpd_T_Cl = get_charge('MPD_T_Cl_1', charges_dict)
        tmc_LD = get_charge('TMC_L_D_1', charges_dict)
        tmc_LP = get_charge('TMC_L_P_1', charges_dict)
        tmc_T0P = get_charge('TMC_T_0P_1', charges_dict)
        tmc_T1P = get_charge('TMC_T_1P_1_2', charges_dict)
        tmc_T2P = get_charge('TMC_T_2P_1', charges_dict)
        tmc_C = get_charge('TMC_C', charges_dict)

        print(f'\nMPD monomers: {n_mpd_mono + n_mpd_mono_Cl + n_mpd_mono_2Cl}')
        print(f'MPD monomer: {n_mpd_mono} with charge: {n_mpd_mono*0:.4f}')
        print(f'MPD monomer-Cl: {n_mpd_mono_Cl} with charge: {n_mpd_mono_Cl*0:.4f}')
        print(f'MPD monomer-2Cl: {n_mpd_mono_2Cl} with charge: {n_mpd_mono_2Cl*0:.4f}')
        print(f'MPD-L: {n_mpd_L} with charge: {n_mpd_L*mpd_L:.4f}')
        print(f'MPD-T: {n_mpd_T} with charge: {n_mpd_T*mpd_T:.4f}')
        print(f'MPD-T-Cl: {n_mpd_T_Cl} with charge: {n_mpd_T_Cl*mpd_T_Cl:.4f}')
        print(f'\nTMC monomers: {n_tmc_mono + n_tmc_mono_0P + n_tmc_mono_1P + n_tmc_mono_2P + n_tmc_mono_3P}')
        print(f'TMC monomer (0P): {n_tmc_mono_0P} with charge: {n_tmc_mono_0P*(-3):.4f}')
        print(f'TMC monomer (1P): {n_tmc_mono_1P} with charge: {n_tmc_mono_1P*(-2):.4f}')
        print(f'TMC monomer (2P): {n_tmc_mono_2P} with charge: {n_tmc_mono_2P*(-1):.4f}')
        print(f'TMC monomer (3P): {n_tmc_mono_3P} with charge: {n_tmc_mono_3P*(0):.4f}')
        print(f'TMC-L-D: {n_tmc_LD} with charge: {n_tmc_LD*tmc_LD:.4f}')
        print(f'TMC-L-P: {n_tmc_LP} with charge: {n_tmc_LP*tmc_LP:.4f}')
        print(f'TMC-T-0P: {n_tmc_T0P} with charge: {n_tmc_T0P*tmc_T0P:.4f}')
        print(f'TMC-T-1P: {n_tmc_T1P} with charge: {n_tmc_T1P*tmc_T1P:.4f}')
        print(f'TMC-T-2P: {n_tmc_T2P} with charge: {n_tmc_T2P*tmc_T2P:.4f}')
        print(f'TMC-C: {n_tmc_C} with charge: {n_tmc_C*tmc_C:.4f}')

        n_deprot = n_tmc_LD + n_tmc_T0P*2 + n_tmc_T1P
        print(f'\nTotal number of deprotonated groups on polymer chains: {n_deprot}')
        print(f'Charge after reassignment: {atoms.charges.sum()}')

        # Set up system of equations to solve for new charges
        # Note: this is solving for the total charge on each fragment, not per-atom charges
        #
        # Let the sum of the charges on a given fragment be represented as follows:
        # MPD_L = m0, MPD_T = m1, MPD_T_Cl = m2, TMC_L_P = t1, TMC_L_D = t2, 
        # TMC_T_0P = t3, TMC_T_1P = t4, TMC_T_2P = t5, TMC_C = t6
        # 
        # m0 = \sum{charges on MPD_L}, which we constrain to be exactly as in OpenEye
        # m0 = 2*t5
        # m1 = m2 + 0.03606 # this difference comes from the OpenEye charges for chlorinated and unchlorinated MPD-T
        # 2*m1 = t1
        # 3*m1 = t6
        # t2 = t1 - 1
        # t3 = t5 - 2
        # t4 = t5 -1
        # Total charge = -n_deprot 
        # 
        # This gives a system of 8 equations and 8 unknowns, which we represent as a matrix equation Ax = b
        # where x = [m1, m2, t1, t2, t3, t4, t5, t6]

        A = np.array([[0,  0,  0, 0, 0, 0,  2,  0],
                    [1, -1,  0, 0, 0, 0,  0,  0],
                    [2,  0, -1, 0, 0, 0,  0,  0],
                    [3,  0,  0, 0, 0, 0,  0, -1],
                    [0,  0, -1, 1, 0, 0,  0,  0],
                    [0,  0,  0, 0, 1, 0, -1,  0],
                    [0,  0,  0, 0, 0, 1, -1,  0],
                    [n_mpd_T, n_mpd_T_Cl, n_tmc_LP, n_tmc_LD, n_tmc_T0P, n_tmc_T1P, n_tmc_T2P, n_tmc_C]])

        b = np.array([[mpd_L],
                    [0.03606],
                    [0],
                    [0],
                    [-1],
                    [-2],
                    [-1],
                    [-n_mpd_L*mpd_L - n_deprot]])

        print('Solving system of equations for new fragment charges...')
        x = np.linalg.solve(A, b)

        # distribute the differences in charge evenly over the 6 aromatic carbons in each fragment
        # dump to a new charges dict: charges_modified.yaml
        diff = {}

        frags = ['MPD_T', 'MPD_T_Cl', 'TMC_L_P', 'TMC_L_D', 'TMC_T_0P', 'TMC_T_1P', 'TMC_T_2P', 'TMC_C']
        orig_charges = [mpd_T, mpd_T_Cl, tmc_LP, tmc_LD, tmc_T0P, tmc_T1P, tmc_T2P, tmc_C]
        for frag, charge, orig in zip(frags,x,orig_charges):
            d = (charge[0]-orig) / 6
            for name in charges_dict:
                if name.startswith(frag):
                    diff[name] = d
            
            print(f'{frag} should have {charge[0]:.5f} charge for a difference of {d*6:.5f} from the original')
            print(f'\tDivided evenly over 6 aromatic carbons: {d:.5f}')

        for frag in charges_dict:
            if frag in diff:
                for atom in charges_dict[frag]:
                    if atom.startswith('ca'):
                        charges_dict[frag][atom]['charge'] = float(round(charges_dict[frag][atom]['charge'],8) + diff[frag])

        with open('charges_modified.yaml', 'w') as f:
            yaml.dump(charges_dict, f)

        

    def hydrolyzed_chlorination(self, target_crosslink_pct=0.8, radius_for_water_search=6, rng_seed=3, 
                                input_topology='PA_ions.top', param_fragment='2MPD_2TMC_1P_Cl', nproc=64, restart_iteration=1, n_counterions=0):
        '''Build hydrolyzed chlorination -- including removing waters, breaking amide bonds, adding Cl and OH groups, updating topology'''

        from gromacs_topology import GromacsTopology

        # save gmx exectutable
        gmx = self.gmx

        n_xlink = len(self.universe.select_atoms(f'type {self.xlink_n_type}'))
        n_possible = len(self.universe.select_atoms(f'type {self.xlink_n_type} {self.term_n_type}'))
        n_to_break = n_xlink - int(target_crosslink_pct * n_possible)

        # track how many ions are added for balancing membrane charge (i.e. how many deprotonated O's have been added)
        # n_counterions = 0

        print(f'Going from {100 * n_xlink / n_possible:.1f}% crosslinking to {100 * (n_xlink - n_to_break) / n_possible:.1f}% crosslinking by breaking {n_to_break} amide bonds')
        print('\n'*5)

        if restart_iteration > 1:
            print('Restarting from iteration', restart_iteration)
            self.tpr = self._generate_tpr(f'final_iter{restart_iteration}.top', f'em3_iter{restart_iteration}.gro', mdp='nvt.mdp', tpr=f'nvt3_iter{restart_iteration}.tpr', gmx=gmx(1))
            self.__init__(f'nvt3_iter{restart_iteration}.gro', frmt='GRO', tpr_file=self.tpr, 
                        xlink_c=self.xlink_c_type, xlink_n=self.xlink_n_type, term_n=self.term_n_type,
                        cl_type=self.cl_type, oh_type=self.oh_type, ow_type=self.ow_type, hw_type=self.hw_type,
                        anion_type=self.anion_type, cation_type=self.cation_type)

            input_topology = f'final_iter{restart_iteration}.top'

        for i in range(restart_iteration,n_to_break):

            print()
            print('-'*50)
            ndashes = 50 - len(f' Iteration {i+1} ')
            print('-'*int(ndashes/2), 'Iteration ', i+1, '-'*int(ndashes/2))
            print('-'*50)

            found_xlink = False
            tries = 0

            while not found_xlink:

                rng = np.random.default_rng(seed=rng_seed+tries)
                all_n = self.universe.select_atoms('type n')
                n = all_n[rng.integers(0, len(all_n))] # randomly select an amide N atom
                c = self.universe.atoms.select_atoms(f'(type c) and (bonded index {n.index})')[0]
                protonate = i < int(n_to_break/2) # protonate half of the COOH groups

            

                #################### REMOVE WATERS #####################

                # remove nearby water molecules to add Cl and OH
                Cl_position = self.remove_water_for_Cl(n, radius_for_water_search=radius_for_water_search, n_waters=2, tol=2)
                if protonate:
                    O_position, H_position = self.remove_water_for_OH(c, Cl_position, radius_for_water_search=radius_for_water_search, protonate=True, n_waters=2, tol=2)
                else:
                    O_position = self.remove_water_for_OH(c, Cl_position, radius_for_water_search=radius_for_water_search, protonate=False, n_waters=2, tol=2)

                tries += 1
                if Cl_position is not None and O_position is not None:
                    found_xlink = True

            print(f'\n\tBreaking bond between N {n} and C {c}')
            if protonate:
                print('\tRemoving waters to make room for Cl and OH groups')
            else:
                print('\tRemoving waters to make room for Cl and O groups')
            print('\tNumber of water molecules before breaking crosslink:', len(self.universe.select_atoms('resname SOL').residues))

            # reassign residue numbers after deleted waters
            n_residues = len(self.atoms.residues)
            resids = np.arange(1, n_residues+1)
            self.universe.add_TopologyAttr('resid', list(resids))
            self.universe.atoms.write('removed_waters.gro')
            print('\tNumber of water molecules after breaking crosslink:', len(self.universe.select_atoms('resname SOL').residues))
            print(f'\tWrote output to removed_waters.gro')

            # update the topology file
            top = GromacsTopology(input_topology, verbose=False)
            water_mol = [mol for mol in top.molecules if mol.name == 'SOL'][0]
            water_mol.n_mols = len(self.universe.select_atoms('resname SOL').residues) # all we have to do is update the number of water molecules
            top.write('removed_waters.top')
            print(f'\tWrote updated topology to removed_waters.top\n\n')

            # reinitialize the class with removed water molecules
            self.tpr = self._generate_tpr('removed_waters.top', 'removed_waters.gro', gmx=gmx(1))
            self.__init__('removed_waters.gro', frmt='GRO', tpr_file=self.tpr, 
                        xlink_c=self.xlink_c_type, xlink_n=self.xlink_n_type, term_n=self.term_n_type,
                        cl_type=self.cl_type, oh_type=self.oh_type, ow_type=self.ow_type, hw_type=self.hw_type,
                        anion_type=self.anion_type, cation_type=self.cation_type)

            #################### BREAK BOND #####################

            print(f'\n\tBreaking amide bond between N {n.index} and C {c.index}')

            # need to get the same N and C atom objects in the new universe
            pos = n.position
            my_ns = self.universe.select_atoms(f'(type n) and (bonded type c) and (point {pos[0]} {pos[1]} {pos[2]} 3)')
            if len(my_ns) != 1:
                raise ValueError(f'{len(my_ns)} amide Ns found near original amide N position', pos)
            
            my_n = my_ns[0]
            my_c = self.universe.atoms.select_atoms(f'(type c) and (bonded index {my_n.index})')[0]

            # delete the bonds and create graphs to determine new molecules
            print('\n\tNumber of polymer bonds before breaking crosslink:', len(self.universe.bonds))
            to_delete = [b for b in my_n.bonds if 'c' in b.atoms.types]
            print('\tDeleting bond(s):', to_delete)
            self.universe.delete_bonds(to_delete)
            self.create_polymer_graphs()
            n_components = [nx.number_connected_components(g) for g in self.polymer_graphs]
            print('\tNumber of polymer bonds after breaking crosslink:', len(self.universe.bonds))
            print('\tNumber of polymer chains after breaking crosslink:', sum(n_components))

            # reassign residue indices
            atom_to_new_resindex, new_resindex_to_atom = self.reassign_residue_indices()
            self.universe.atoms.write('broken_bonds.gro')
            print('\tWrote output to broken_bonds.gro')

            # update the topology file
            # Note: my GromacsTopology class writes out only using the molecules, so it should be fine if we update all the molecule information
            print('\n\tUpdating topology file to reflect broken amide bond')
            top = GromacsTopology('removed_waters.top', verbose=False)
            top = self.update_topology_after_breaking_bond(top, my_n, my_c, new_resindex_to_atom, atom_to_new_resindex)
            top.write('broken_bonds.top')
            print(f'\tWrote updated topology to broken_bonds.top')

            # reassign global atom indices to the GRO file
            tmp_u = mda.Universe('broken_bonds.gro')
            global_map = self.get_global_atom_indices(tmp_u, top, breaking_bonds=True, n_counterions=n_counterions)
            tmp_u.add_TopologyAttr('ids', [global_map[atom.ix] for atom in tmp_u.atoms])
            tmp_u.atoms.sort('ids').write('broken_bonds_sorted.gro')
            print('\tWrote sorted output to broken_bonds_sorted.gro\n\n')

            # reinitialize the class with broken bonds
            self.tpr = self._generate_tpr('broken_bonds.top', 'broken_bonds_sorted.gro', gmx=gmx(1))
            self.__init__('broken_bonds_sorted.gro', frmt='GRO', tpr_file=self.tpr, 
                        xlink_c=self.xlink_c_type, xlink_n=self.xlink_n_type, term_n=self.term_n_type,
                        cl_type=self.cl_type, oh_type=self.oh_type, ow_type=self.ow_type, hw_type=self.hw_type,
                        anion_type=self.anion_type, cation_type=self.cation_type)


            #################### ADD Cl and O(H) #####################

            print(f'\n\tAdding Cl and O(H) groups to positions previously occupied by waters')

            # need to get the same N and C atom objects in the new universe
            pos = n.position
            my_ns = self.universe.select_atoms(f'(type n) and (not bonded type c) and (point {pos[0]} {pos[1]} {pos[2]} 1)')
            if len(my_ns) != 1:
                raise ValueError(f'{len(my_ns)} amide Ns found near original amide N position', pos)
            
            pos = c.position
            my_Cs = self.universe.select_atoms(f'(type c) and (not bonded type n) and (point {pos[0]} {pos[1]} {pos[2]} 1)')
            if len(my_Cs) != 1:
                raise ValueError(f'{len(my_Cs)} broken amide Cs found near original amide C position', pos)

            my_n = my_ns[0]
            my_c = my_Cs[0]

            # add new atoms to GRO file
            if protonate:
                merged_u = self.add_new_atoms_for_chlorination(my_n, my_c, Cl_position, O_position, H_position=H_position)
            else:
                merged_u = self.add_new_atoms_for_chlorination(my_n, my_c, Cl_position, O_position)

            merged_u.atoms.write('added_atoms.gro')
            print('\tWrote output to added_atoms.gro')

            # Get bonded parameters for new bonds from the parameter topology
            param_gro = PolymAnalysis(f'{param_fragment}_box.gro', frmt='GRO', tpr_file=f'{param_fragment}_converted.tpr', 
                                    xlink_c=self.xlink_c_type, xlink_n=self.xlink_n_type, term_n=self.term_n_type,
                                    cl_type=self.cl_type, oh_type=self.oh_type, ow_type=self.ow_type, hw_type=self.hw_type,
                                    anion_type=self.anion_type, cation_type=self.cation_type)
            param_top = GromacsTopology(f'{param_fragment}_converted.top')

            # update the topology file with new atoms and bonded parameters
            top = GromacsTopology('broken_bonds.top', verbose=False)
            top, total_atoms = self.update_topology_with_Cl(top, my_n, param_gro, param_top)

            if protonate:  # only protonated half of the COOH groups
                top, total_atoms = self.update_topology_with_O(top, my_c, total_atoms, param_gro, param_top, protonate=True)
                top, total_atoms = self.update_topology_with_H(top, my_c, total_atoms, param_gro, param_top)
            else:
                top, total_atoms = self.update_topology_with_O(top, my_c, total_atoms, param_gro, param_top, protonate=False)
            
            top.write('added_atoms.top')
            print(f'\tWrote updated topology to added_atoms.top')

            # reassign global atom indices to the GRO file
            tmp_u = mda.Universe('added_atoms.gro')
            global_map = self.get_global_atom_indices(tmp_u, top, breaking_bonds=False, n_counterions=n_counterions)
            tmp_u.add_TopologyAttr('ids', [global_map[atom.ix] for atom in tmp_u.atoms])
            tmp_u.atoms.sort('ids').write('added_atoms_sorted.gro')
            print('\tWrote sorted output to added_atoms_sorted.gro')

            # reinitialize the class with added chlorination atoms
            self.tpr = self._generate_tpr('added_atoms.top', 'added_atoms_sorted.gro', flags={'maxwarn' : 2}, gmx=gmx(1))
            self.__init__('added_atoms_sorted.gro', frmt='GRO', tpr_file=self.tpr, 
                        xlink_c=self.xlink_c_type, xlink_n=self.xlink_n_type, term_n=self.term_n_type,
                        cl_type=self.cl_type, oh_type=self.oh_type, ow_type=self.ow_type, hw_type=self.hw_type,
                        anion_type=self.anion_type, cation_type=self.cation_type)

            #################### REASSIGN CHARGES #####################

            print(f'\n\tReassigning charges after chlorination')

            top = GromacsTopology('added_atoms.top', verbose=False)
            self.calculate_charges_after_chlorination() # this creates a charges_modified.yaml file that has the new charges to assign
            top = self.reassign_charges_after_chlorination(top)
            top.write('recharged.top')
            print(f'\tWrote recharged topology to recharged.top')

            # reinitialize the class with correct charges on polymer
            self.tpr = self._generate_tpr('recharged.top', 'added_atoms_sorted.gro', flags={'maxwarn' : 1}, gmx=gmx(1))
            self.__init__('added_atoms_sorted.gro', frmt='GRO', tpr_file=self.tpr, 
                        xlink_c=self.xlink_c_type, xlink_n=self.xlink_n_type, term_n=self.term_n_type,
                        cl_type=self.cl_type, oh_type=self.oh_type, ow_type=self.ow_type, hw_type=self.hw_type,
                        anion_type=self.anion_type, cation_type=self.cation_type)

            if protonate:
                o_type = 'oh'
            else:
                o_type = 'o'

            check_around = 1
            added_atom = self.universe.select_atoms(f'(type {o_type}) and (point {O_position[0]} {O_position[1]} {O_position[2]} {check_around})') 
            while len(added_atom) == 0: # if we cannot find the O within the default tolerance, check with increasing cutoff radii
                check_around += 0.5
                added_atom = self.universe.select_atoms(f'(type {o_type}) and (point {O_position[0]} {O_position[1]} {O_position[2]} {check_around})')
            print(f'\n\tAdded O atom: {added_atom[0]} (which was within {check_around} of the original position)')
            dists = distances.distance_array(added_atom, self.universe.atoms-added_atom, box=self.universe.dimensions)
            print(f'\tClosest distances to other atoms: {dists[dists < 3]}')

            # _run(f'cp recharged.top final_iter{i+1}.top')
            _run(['cp', 'recharged.top', f'final_iter{i+1}.top'])

            #################### ADD ADDITIONAL IONS #####################

            if not protonate:
                print(f'\n\tAdding additional ions to neutralize system after chlorination')
                self.insert_ions(n_cations=1, cation_name='NA', cation_charge=1, top=f'final_iter{i+1}.top', output=f'final_iter{i+1}.gro', gmx=gmx(1))
                
                # rename NA ions in GRO file to Na to match topology
                tmp_u = mda.Universe(f'final_iter{i+1}.gro')
                na_ions = tmp_u.select_atoms('resname NA')
                
                new_names = []
                for atom in tmp_u.atoms:
                    if atom in na_ions:
                        new_names.append('Na')
                    else:
                        new_names.append(atom.name)
                
                tmp_u.add_TopologyAttr('names', new_names)
                tmp_u.atoms.write(f'final_iter{i+1}.gro')

                n_counterions += 1  # track how many counterions have been added for balancing membrane charge

            
            else:
                # _run(f'cp added_atoms_sorted.gro final_iter{i+1}.gro')
                _run(['cp', 'added_atoms_sorted.gro', f'final_iter{i+1}.gro'])

            #################### RUN MINIMIZATION #####################
            
            print(f'\n\tRunning energy minimization after chlorination')

            # need to get the same N and C atom objects in the new universe
            pos = n.position
            my_ns = self.universe.select_atoms(f'(type nh) and (bonded type cl) and (point {pos[0]} {pos[1]} {pos[2]} 2)')
            if len(my_ns) != 1:
                raise ValueError(f'{len(my_ns)} broken amide Ns found near original amide N position', pos)
            
            pos = c.position
            my_Cs = self.universe.select_atoms(f'(type c) and (not bonded type n) and (point {pos[0]} {pos[1]} {pos[2]} 2)')
            if len(my_Cs) != 1:
                raise ValueError(f'{len(my_Cs)} broken amide Cs found near original amide C position', pos)

            my_n = my_ns[0]
            my_c = my_Cs[0]

            # turn off Cl and OH nonbonded interactions by setting their epsilon to zero and charge to zero
            top = GromacsTopology(f'final_iter{i+1}.top', verbose=False)

            top_c = top.atoms[my_c.index]
            top_n = top.atoms[my_n.index]

            # create new atom types with epsilon = 0
            added_types = ['oh', 'ho', 'o', 'c', 'nh', 'cl']
            for atype in added_types:
                top.atomtypes[f'{atype}_off'] = top.atomtypes[atype].copy()
                top.atomtypes[f'{atype}_off'].type = f'{atype}_off'
                top.atomtypes[f'{atype}_off'].bondingtype = f'{atype}_off'
                top.atomtypes[f'{atype}_off'].epsilon = 0

            # turn off for c, o, oh, ho
            top_c.type = top_c.type + '_off'
            top_c.charge = 0
            for bond in top_c.bonds:
                for a in bond.atoms:
                    if a.type in ['c_off', 'ca']:
                        continue

                    a.type = a.type + '_off'
                    a.charge = 0

                    if a.type == 'oh_off':
                        oh_ho_bond = [b for b in a.bonds if 'ho' in [b.atoms[0].type, b.atoms[1].type]][0]
                        ho_atom = [atom for atom in oh_ho_bond.atoms if atom.type == 'ho'][0]
                        ho_atom.type = ho_atom.type + '_off'
                        ho_atom.charge = 0
            
            # turn off for n, cl
            top_n.type = top_n.type + '_off'
            top_n.charge = 0
            n_cl_bond = [b for b in top_n.bonds if 'cl' in [b.atoms[0].type, b.atoms[1].type]][0]
            cl_atom = [atom for atom in n_cl_bond.atoms if atom.type == 'cl'][0]
            cl_atom.type = cl_atom.type + '_off'
            cl_atom.charge = 0

            top.write(f'final_iter{i+1}_nonbonded_off.top')

            self.tpr = self._generate_tpr(f'final_iter{i+1}_nonbonded_off.top', f'final_iter{i+1}.gro', mdp='min.mdp', tpr=f'em1_iter{i+1}.tpr', gmx=gmx(1), flags={'maxwarn' : 1}) # note this creates a charged system so need maxwarn
            # cmd = [f'{gmx(nproc)} mdrun -s {self.tpr} -deffnm em1_iter{i+1}']
            cmd = gmx(nproc) + ['mdrun', '-s', self.tpr, '-deffnm', f'em1_iter{i+1}']
            _run(cmd)

            self.tpr = self._generate_tpr(f'final_iter{i+1}_nonbonded_off.top', f'em1_iter{i+1}.gro', mdp='nvt.mdp', tpr=f'nvt1_iter{i+1}.tpr', gmx=gmx(1), flags={'maxwarn' : 1}) 
            # cmd = [f'{gmx(nproc)} mdrun -s {self.tpr} -deffnm nvt1_iter{i+1}']
            cmd = gmx(nproc) + ['mdrun', '-s', self.tpr, '-deffnm', f'nvt1_iter{i+1}']
            _run(cmd)

            # turn vdW interactions back on
            top = GromacsTopology(f'final_iter{i+1}.top', verbose=False)

            top_c = top.atoms[my_c.index]
            top_n = top.atoms[my_n.index]

            # turn off charge for c, o, oh, ho
            top_c.charge = 0
            for bond in top_c.bonds:
                for a in bond.atoms:
                    if a.type in ['c', 'ca']:
                        continue

                    a.charge = 0

                    if a.type == 'oh':
                        oh_ho_bond = [b for b in a.bonds if 'ho' in [b.atoms[0].type, b.atoms[1].type]][0]
                        ho_atom = [atom for atom in oh_ho_bond.atoms if atom.type == 'ho'][0]
                        ho_atom.charge = 0
            
            # turn off charge for n, cl
            top_n.charge = 0
            n_cl_bond = [b for b in top_n.bonds if 'cl' in [b.atoms[0].type, b.atoms[1].type]][0]
            cl_atom = [atom for atom in n_cl_bond.atoms if atom.type == 'cl'][0]
            cl_atom.charge = 0

            top.write(f'final_iter{i+1}_charges_off.top')

            self.tpr = self._generate_tpr(f'final_iter{i+1}_charges_off.top', f'nvt1_iter{i+1}.gro', mdp='min.mdp', tpr=f'em2_iter{i+1}.tpr', gmx=gmx(1), flags={'maxwarn' : 1}) 
            # cmd = [f'{gmx(nproc)} mdrun -s {self.tpr} -deffnm em2_iter{i+1}']
            cmd = gmx(nproc) + ['mdrun', '-s', self.tpr, '-deffnm', f'em2_iter{i+1}']
            _run(cmd)

            self.tpr = self._generate_tpr(f'final_iter{i+1}_charges_off.top', f'em2_iter{i+1}.gro', mdp='nvt.mdp', tpr=f'nvt2_iter{i+1}.tpr', gmx=gmx(1), flags={'maxwarn' : 1})
            # cmd = [f'{gmx(nproc)} mdrun -s {self.tpr} -deffnm nvt2_iter{i+1}']
            cmd = gmx(nproc) + ['mdrun', '-s', self.tpr, '-deffnm', f'nvt2_iter{i+1}']
            _run(cmd)

            # turn charges back on
            self.tpr = self._generate_tpr(f'final_iter{i+1}.top', f'nvt2_iter{i+1}.gro', mdp='min.mdp', tpr=f'em3_iter{i+1}.tpr', gmx=gmx(1), flags={'maxwarn' : 1}) 
            # cmd = [f'{gmx(nproc)} mdrun -s {self.tpr} -deffnm em3_iter{i+1}']
            cmd = gmx(nproc) + ['mdrun', '-s', self.tpr, '-deffnm', f'em3_iter{i+1}']
            _run(cmd)

            self.tpr = self._generate_tpr(f'final_iter{i+1}.top', f'em3_iter{i+1}.gro', mdp='nvt.mdp', tpr=f'nvt3_iter{i+1}.tpr', gmx=gmx(1)) 
            # cmd = [f'{gmx(nproc)} mdrun -s {self.tpr} -deffnm nvt3_iter{i+1}']
            cmd = gmx(nproc) + ['mdrun', '-s', self.tpr, '-deffnm', f'nvt3_iter{i+1}']
            _run(cmd)

            # reinitialize the class with minimized structure
            self.__init__(f'nvt3_iter{i+1}.gro', frmt='GRO', tpr_file=self.tpr, 
                        xlink_c=self.xlink_c_type, xlink_n=self.xlink_n_type, term_n=self.term_n_type,
                        cl_type=self.cl_type, oh_type=self.oh_type, ow_type=self.ow_type, hw_type=self.hw_type,
                        anion_type=self.anion_type, cation_type=self.cation_type)

            if (i+1) % 10 == 0: # every 10 iterations, run NPT to relax box
                self.tpr = self._generate_tpr(f'final_iter{i+1}.top', f'nvt3_iter{i+1}.gro', mdp='npt.mdp', tpr=f'npt_iter{i+1}.tpr', gmx=gmx(1))
                # cmd = [f'{gmx(nproc)} mdrun -s {self.tpr} -deffnm npt_iter{i+1}']
                cmd = gmx(nproc) + ['mdrun', '-s', self.tpr, '-deffnm', f'npt_iter{i+1}']
                _run(cmd)

                # reinitialize the class with minimized structure
                self.__init__(f'npt_iter{i+1}.gro', frmt='GRO', tpr_file=self.tpr, 
                            xlink_c=self.xlink_c_type, xlink_n=self.xlink_n_type, term_n=self.term_n_type,
                            cl_type=self.cl_type, oh_type=self.oh_type, ow_type=self.ow_type, hw_type=self.hw_type,
                            anion_type=self.anion_type, cation_type=self.cation_type)

            input_topology = f'final_iter{i+1}.top'
            self._run('rm *#*') # remove backups
            # _run(['rm', '*#*'])

            print('-'*50)
            print('\n'*5)
            

    def insert_ions(self, n_anions=0, anion_name=None, anion_charge=-1, n_cations=0, cation_name=None, cation_charge=1, top='PA_ions.top', output='PA_ions.gro', gmx=None):
        '''Insert ions with gmx genion'''

        if gmx is None:
            gmx = self.gmx

        if isinstance(gmx, list): # convert list to string
            gmx = ' '.join(gmx)

        # first, write an ndx file with the water selection
        waters = self.universe.select_atoms(f'type {self.ow_type} {self.hw_type}')
        waters.write('waters.ndx', name='SOL')

        if cation_name is None and anion_name is None:
            raise TypeError('No ions given. Please provide an anion, cation, or both.')
        elif cation_name is None:
            cmd = f'{gmx} genion -s {self.tpr} -p {top} -o {output} -nn {n_anions} -nname {anion_name} -nq {anion_charge} -n waters.ndx'
        elif anion_name is None:
            cmd = f'{gmx} genion -s {self.tpr} -p {top} -o {output} -np {n_cations} -pname {cation_name} -pq {cation_charge} -n waters.ndx'
        else:
            cmd = f'{gmx} genion -s {self.tpr} -p {top} -o {output} -nn {n_anions} -nname {anion_name} -nq {anion_charge} -np {n_cations} -pname {cation_name} -pq {cation_charge} -n waters.ndx'

        print(cmd)
        self._run(cmd)

        # reinitialize the class with updated ions
        self._generate_tpr(top, output, tpr=self.tpr, gmx=gmx.split(), flags={'maxwarn' : 1})
        self.__init__(output, frmt='GRO', tpr_file=self.tpr, 
                      xlink_c=self.xlink_c_type, xlink_n=self.xlink_n_type, term_n=self.term_n_type,
                      cl_type=self.cl_type, oh_type=self.oh_type, ow_type=self.ow_type, hw_type=self.hw_type,
                      anion_type=self.anion_type, cation_type=self.cation_type)

    def calculate_density(self, atom_group='all', box=False):
        '''Calculate density of a selection of the polymer membrane'''

        if isinstance(atom_group, str): # if provided selection language, make AtomGroup
            g = self.universe.select_atoms(atom_group)
        else: # else assume input is AtomGroup
            g = atom_group

        if box:
            xlo, xhi = self.box[:,0]
            ylo, yhi = self.box[:,1]
            zlo, zhi = self.box[:,2]
        else:
            xlo, xhi = g.positions[:,0].min(), g.positions[:,0].max() # NOTE: these limits will be slightly different than self.box
            ylo, yhi = g.positions[:,1].min(), g.positions[:,1].max() #       since it is using the min and max atom coordinates, but
            zlo, zhi = g.positions[:,2].min(), g.positions[:,2].max() #       the difference should only be at most 0.002

        total_mass = g.masses.sum() / 6.022 / 10**23 # [g/mol * mol/# = g]
        s1 = (xhi-xlo) * 10**-8 # [Ang * 10^8 cm/Ang = cm]
        s2 = (yhi-ylo) * 10**-8 # cm
        s3 = (zhi-zlo) * 10**-8 # cm
        vol = s1*s2*s3 # cm^3
        density = total_mass / vol

        if box:
            print('Density of PA membrane: {:.4f} g/cm^3'.format(density))
        else:
            print('Density of {}: {:.4f} g/cm^3'.format(atom_group, density) )

        return density
    

    def atom_distribution(self, percentage=False):
        '''Plot distribution of atoms over the resids'''

        X = []
        Y = []

        for res in self.universe.residues:
            X.append(res.resid)
            Y.append(len(res.atoms))

        X = np.array(X)
        Y = np.array(Y)
        print(f'Resid {Y.argmax()+1} is the largest chain with {Y.max()} atoms')
        if percentage:
            Y = Y / self.n_atoms * 100

        self.atom_distribution = (X, Y)
        print('Atom distribution saved in self.atom_distribution')

        plt.plot(X,Y)
        plt.xlabel('resid')
        plt.ylabel('number of atoms')
        if percentage:
            plt.ylabel('percentage of atoms')
        plt.savefig('atom_distribution.png')
        plt.show()
        plt.close()


    def plot_crosslinked(self):
        '''Plot crosslinking percentage as a function of simulation time'''

        if len(self.logs) == 0:
            raise AttributeError('No log files loaded... Try load_log() first')

        self.possible_bonds = self.n_monomers[0]*2
        print('Using ({} MPD molecules)*(2 possible bonds) = {} total possible bonds for crosslinking percentage calculation'.format(self.n_monomers[0], self.possible_bonds))

        X = []
        Y = []
        old_bonds = 0
        old_steps = 0
        for current_log in self.logs:
        
            for l, log in enumerate(current_log.partial_logs): 

                if 'f_fxrct[1]' in log.keys():
                    steps = log.get('Step')
                    bonds = log.get('f_fxrct[1]')

                    for x in steps:
                        X.append((old_steps + x)*0.001) # multiply each step by time (0.001 ps per step)

                    for y in bonds:
                        Y.append(old_bonds + y)

                    old_bonds += bonds[-1]
                    old_steps += steps[-1]

        self.bonds_formed = np.array(Y)
        self.percent_crosslinked = np.array(Y) / self.possible_bonds * 100

        X = np.array(X)
        Y = np.array(Y)

        fig, ax = plt.subplots(1,1, figsize=(10,10))

        print('Final percent crosslinked:', Y[-1] / self.possible_bonds * 100)
        plt.plot(X, Y / self.possible_bonds * 100)
        plt.xlabel('Time (ps)', fontsize=16)
        plt.ylabel('Percent crosslinked', fontsize=16)
        plt.ylim(0,100)
        plt.xlim(-500,)
        plt.xticks(fontsize=16, rotation=45)
        plt.yticks(fontsize=16)

        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.xaxis.set_minor_locator(MultipleLocator(250))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(1))

        ax.xaxis.set_major_formatter('{x:,.0f}')

        plt.savefig('percent_crosslinked.png')
        plt.show()
        plt.close()


    def plot_ts(self, log, y, timestep=0.001, start_log=0, end_log=None, show_legend=False, plot_end=False):
        '''Plot a timeseries of a given thermodynamic quantity'''

        logfile = log

        if end_log is None:
            plogs = logfile.partial_logs[start_log:]
        else:
            plogs = logfile.partial_logs[start_log:end_log+1]

        fig, ax = plt.subplots(1,1)

        n = 0
        for plog in plogs:

            time = plog.get('Step')*timestep

            if type(y) is list:
                
                for i in y:

                    if not i in plog.keys():
                        raise TypeError('{} not available in partial log {}. Try one of {}'.format(i, n, plog.keys()))
            
                    Y = plog.get(i)
                    plt.plot(time, Y, label=i + ': partial log {}'.format(n))
            
            else:

                if not y in plog.keys():
                    raise TypeError('{} not available in partial log {}. Try one of {}'.format(y, n, plog.keys()))
            
                Y = plog.get(y)
                plt.plot(time, Y, label=y + ': partial log {}'.format(n))


            n += 1

        if show_legend:
            plt.legend()

        if plot_end:
            plt.xlim(time[-100], time[-1])
            plt.ylim(Y[-1]*0.9, Y[-1]*1.1)

        plt.xlabel('Time (ps)')
        plt.ylabel(y)
        plt.savefig('{}_vs_time.png'.format(y))
        plt.show()
        plt.close()


    def density_profile(self, atom_group, bin_width=0.5, dim='z', method='atom', frameby=1):
        '''Calculate the partial density across the box'''

        if isinstance(atom_group, str): # if provided selection language, make AtomGroup
            ag = self.universe.select_atoms(atom_group)
        else: # else assume input is AtomGroup
            ag = atom_group

        print(f'\nCalculating the partial density profile of {atom_group} in the {dim} dimension...')

        dims = {'x': 0, 'y': 1, 'z': 2}
        d = dims[dim]
        box = self.universe.dimensions[d]

        n_bins = int(box / bin_width)
        bins = np.linspace(self.box[0,d], self.box[0,d] + box, num=n_bins)

        counts = np.zeros(n_bins-1)
        
        if self.trajectory is None:
            for b in tqdm(range(n_bins-1)):
                lb = bins[b]
                ub = bins[b+1]
                bin_atoms = self.universe.select_atoms(f'prop {dim} > {lb} and prop {dim} < {ub} and group ag', ag=ag)
                if method in ['atom', 'atoms', 'all']:
                    counts[b] += len(bin_atoms)
                elif method in ['molecule', 'mol', 'residue', 'res']: 
                    counts[b] += bin_atoms.n_residues
                elif method in ['mass', 'mass density']:
                    box_dims = [self.box[1,i] - self.box[0,i] for i in range(3) if i != d]
                    dV = box_dims[0] * box_dims[1] * (ub-lb) * (10**-8)**3
                    mass = bin_atoms.masses.sum() / 6.022 / 10**23
                    counts[b] += mass / dV
                elif method in ['charge', 'charge density']:
                    box_dims = [self.box[1,i] - self.box[0,i] for i in range(3) if i != d]
                    dV = box_dims[0] * box_dims[1] * (ub-lb) * (10**-8)**3
                    charge = bin_atoms.charges.sum() / 6.022 / 10**23
                    counts[b] += charge / dV
        else:
            for ts in tqdm(self.trajectory[::frameby]):
                self.box = self._get_box()
                for b in range(n_bins-1):
                    lb = bins[b]
                    ub = bins[b+1]
                    bin_atoms = self.universe.select_atoms(f'prop {dim} > {lb} and prop {dim} < {ub} and group ag', ag=ag)

                    if method in ['atom', 'atoms', 'all', 'number']:
                        counts[b] += len(bin_atoms)
                    elif method in ['molecule', 'mol', 'residue', 'res']: 
                        counts[b] += bin_atoms.n_residues
                    elif method in ['mass', 'mass density']:
                        box_dims = [self.box[1,i] - self.box[0,i] for i in range(3) if i != d]
                        dV = box_dims[0] * box_dims[1] * (ub-lb) * (10**-8)**3
                        mass = bin_atoms.masses.sum() / 6.022 / 10**23
                        counts[b] += mass / dV
                    elif method in ['charge', 'charge density']:
                        box_dims = [self.box[1,i] - self.box[0,i] for i in range(3) if i != d]
                        dV = box_dims[0] * box_dims[1] * (ub-lb) * (10**-8)**3
                        charge = bin_atoms.charges.sum() / 6.022 / 10**23
                        counts[b] += charge / dV

            counts = counts / len(self.trajectory[::frameby])

        return bins, counts

    def crosslinked_profile(self, bin_width=5, dim='z'):
        '''Calculate the partial crosslinking density across the box'''

        dims = {'x': 0, 'y': 1, 'z': 2}
        d = dims[dim]
        box = self.universe.dimensions[d]

        n_bins = int(box / bin_width)
        bins = np.linspace(self.box[0,d], self.box[0,d] + box, num=n_bins)

        self.n_xlinks = np.zeros(n_bins-1)
        for b in tqdm(range(n_bins-1)):
            lb = bins[b]
            ub = bins[b+1]

            self.n_xlinks[b] = len(self.universe.select_atoms('prop {} > {} and prop {} < {} and {} {} and bonded {} {}'.format(dim, lb, dim, ub, self.atsel, self.xlink_c_type, self.atsel, self.xlink_n_type)))

        fig, ax = plt.subplots(1,1, figsize=(8,8))
        plt.plot(bins[:-1], self.n_xlinks)
        plt.xlabel('{} coordinate'.format(dim))
        plt.ylabel('number of crosslinks')
        plt.savefig('crosslinked_profile.png')
        plt.show()
        plt.close()


    def terminated_profile(self, bin_width=5, dim='z', count_type='Cl'):
        '''Calculated the (un)terminated density across the box'''

        if count_type in ['Cl', 'CL', 'cl']:
            print('Counting remaining Cl in system...')
            term_sel = 'type {}'.format(self.cl_type)
            term_label = 'remaining Cl'
        elif count_type in ['O', 'OH', 'o', 'oh']:
            print('Counting hydroxyl groups in system...')
            term_sel = 'type {}'.format(self.oh_type)
            term_label = 'OH groups'
        elif count_type in ['bonded O', 'bonded OH', 'bonded o', 'bonded oh', 'term C', 'terminated C']:
            print('Counting bonded hydroxyl groups in system...')
            term_sel = 'type {} and bonded type {}'.format(self.oh_type, self.xlink_c_type)
            term_label = 'bonded OH groups'
        elif count_type in ['free O', 'free OH', 'free o', 'free oh', 'nonbonded OH']:
            print('Counting free hydroxyl groups in system...')
            term_sel = 'type {} and not bonded type {}'.format(self.oh_type, self.xlink_c_type)
            term_label = 'free OH groups'
        else:
            raise TypeError('count_type = {} not implemented. Try Cl, OH, free OH, or bonded OH'.format(count_type))

        dims = {'x': 0, 'y': 1, 'z': 2}
        d = dims[dim]
        box = self.universe.dimensions[d]

        n_bins = int(box / bin_width)
        bins = np.linspace(self.box[0,d], self.box[0,d] + box, num=n_bins)
        self.term_bins = bins

        self.n_term = np.zeros(n_bins-1)
        for b in tqdm(range(n_bins-1)):
            lb = bins[b]
            ub = bins[b+1]

            self.n_term[b] = len(self.universe.select_atoms('prop {} > {} and prop {} < {} and {}'.format(dim, lb, dim, ub, term_sel)))

        fig, ax = plt.subplots(1,1, figsize=(8,8))
        plt.plot(bins[:-1], self.n_term)
        plt.xlabel('{} coordinate'.format(dim))
        plt.ylabel('number of {}'.format(term_label))
        plt.savefig('terminated_profile_{}.png'.format(term_label))
        plt.show()
        plt.close()


    def rdf(self, atom_group1, atom_group2, range=(0,15), nbins=75, output='rdf.png', **kwargs):
        '''Calculate the RDF from atom_group1 to atom_group2 using MDAnalysis InterRDF'''

        from MDAnalysis.analysis.rdf import InterRDF

        print(f'Calculating the RDF for {atom_group1} and {atom_group2} over {self.universe.trajectory} frames')

        if isinstance(atom_group1, str): # if provided selection language, make AtomGroup
            g1 = self.universe.select_atoms(atom_group1)
        else: # else assume input is AtomGroup
            g1 = atom_group1

        if isinstance(atom_group2, str):
            g2 = self.universe.select_atoms(atom_group2)
        else:
            g2 = atom_group2

        rdf = InterRDF(g1, g2, range=range, nbins=nbins, verbose=True)
        rdf.run(**kwargs)
        self.rdf_results = rdf.results

        fig, ax = plt.subplots(1,1, figsize=(8,8))
        plt.plot(rdf.results.bins, rdf.results.rdf)
        plt.xlabel('r (A)')
        plt.ylabel('g(r)')
        plt.savefig(output)
        plt.show()
        plt.close()


    def plot_membrane(self, PA, ydim='x', water=None, weight='mass', lb=None, ub=None, frame=-1, 
                      ax=None, savename=None, pa_kwargs={}, water_kwargs={}):
        '''
        Plot a KDE plot of the membrane
        
        Parameters
        ----------
        PA : MDAnalysis AtomGroup
            AtomGroup of the polymer membrane
        ydim : str
            Dimension to plot on the y axis ('x' or 'y')
        water : MDAnalysis AtomGroup
            AtomGroup of the water to plot (optional)
        weight : str
            Weighting for the polymer ('mass' or 'charge')
        lb : float
            Lower bound of the membrane (optional)
        ub : float
            Upper bound of the membrane (optional)
        frame : int
            Frame to plot (optional)
        savename : str
            Name of the file to save the plot (optional)
        pa_kwargs : dict
            Keyword arguments to pass to seaborn kdeplot for polymer
        water_kwargs : dict
            Keyword arguments to pass to seaborn kdeplot for water

        Returns
        -------
        ax : matplotlib Axes
            Axes object of the plot

        '''

        self.box = self._get_box()

        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(8,4))

        if ydim == 'x':
            d = 0
        elif ydim == 'y':
            d = 1

        self.universe.trajectory[frame]
        coords = pd.DataFrame(PA.positions, columns=['x', 'y', 'z'])

        if weight == 'mass':
            weights = PA.masses
        if weight == 'charge':
            weights = np.abs(PA.charges)
        
        # plot the polymer with a kernel density estimate
        sns.kdeplot(coords, x='z', y=ydim, cmap='Greys', ax=ax, fill=True, 
                    levels=10, thresh=0.05, cut=0, weights=weights, **pa_kwargs)

        if water is not None:
            if 'fill' not in water_kwargs:
                water_kwargs['fill'] = False
            if 'levels' not in water_kwargs:
                water_kwargs['levels'] = 10
            if 'thresh' not in water_kwargs:
                water_kwargs['thresh'] = 0.05
            if 'cut' not in water_kwargs:
                water_kwargs['cut'] = 0

            coords = pd.DataFrame(water.positions, columns=['x', 'y', 'z'])
            wrapped_z_below = coords['z'] - (self.box[1,2] - self.box[0,2])
            wrapped_z_above = coords['z'] + (self.box[1,2] - self.box[0,2])
            wrapped_coords_below = coords.copy()
            wrapped_coords_below['z'] = wrapped_z_below
            wrapped_coords_above = coords.copy()
            wrapped_coords_above['z'] = wrapped_z_above

            coords = pd.concat([coords, wrapped_coords_below, wrapped_coords_above], ignore_index=True)
    
            # plot the water with a kernel density estimate
            sns.kdeplot(coords, x='z', y=ydim, cmap='Blues', ax=ax, **water_kwargs)
            
        # plot the periodic boundaries
        # p = plt.Rectangle((self.box[0,2],self.box[0,d]), self.box[1,2]-self.box[0,2], self.box[1,d]-self.box[0,d], fill=False, lw=1)
        # ax.add_patch(p)

        # Plot bulk membrane cutoffs
        if lb is not None:
            ax.axhline(lb, c='r', ls='dashed')
            ax.axhline(lb, c='r', ls='dashed', label='bulk boundaries')
        if ub is not None:
            ax.axhline(ub, c='r', ls='dashed')
            ax.axhline(ub, c='r', ls='dashed')

        # formatting
        ax.set_xlabel('z dimension ($\AA$)')
        ax.set_ylabel(ydim+' dimension ($\AA$)')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(self.box[0,2], self.box[1,2])
        ax.set_ylim(self.box[0,d], self.box[1,d])
        
        if savename is not None:
            fig.savefig(savename)

        return ax


    def write_data(self, output, frame=-1):
        '''Write new LAMMPS data file with (possibly) new coordinates and charges'''

        # Open output file and get original data file
        out = open(output, 'w')
        lines = self.original
        box = self.box

        # Write header
        header = []
        header.append(lines.pop(0))
        header.append('\n')

        top_types = ['atom', 'bond', 'angle', 'dihedral', 'improper']
        for top_type in top_types:
            line = '{} {}s\n'.format(getattr(self, 'n_{}s'.format(top_type)), top_type)
            header.append(line)

            line = '{} {} types\n'.format(getattr(self, 'n_{}_types'.format(top_type)), top_type)
            header.append(line)
        
        print('WARNING: PolymAnalysis.write_data() only writes cubic boxes')

        header.append("""
        {:.18g} {:.18g} xlo xhi
        {:.18g} {:.18g} ylo yhi
        {:.18g} {:.18g} zlo zhi
        """.format(box[0,0], box[1,0], box[0,1], box[1,1], box[0,2], box[1,2]))

        header.append('\nMasses\n\n')
        for at in self.masses:
            line = '{} {}\n'.format(at, self.masses[at])
            header.append(line)

        print('WARNING: PolymAnalysis.write_data() does not include any parameters (e.g. Pair Coefs, Bond Coefs)')

        # Write the atom coordinates
        print('Writing atom coordinates and charges...')
        header.append('\nAtoms # full\n\n')
        xyz = self.coords[frame,:,:]
        charges = self.charges

        # Get index of where Atoms section begins
        if 'Atoms\n' in lines:
            idx = lines.index('Atoms\n') + 2
        elif 'Atoms # full\n' in lines:
            idx = lines.index('Atoms # full\n') + 2
        else:
            raise SyntaxError('Could not locate Atoms section in {}'.format(self.filename))
        
        for n in tqdm(range(self.n_atoms)):
            atom = lines[idx+n].split()
            a_id = int(atom[0])
            mol_id = int(atom[1])
            atype = int(atom[2])
            charge = charges[a_id-1]
            x, y, z = xyz[a_id-1,:]

            line = '{:.0f} {:.0f} {:.0f} {:.8g} {:.18g} {:.18g} {:.18g}\n'.format(a_id, mol_id, atype, charge, x, y, z)
            header.append(line)
        
        # Write all other information unchanged
        print('WARNING: PolymAnalysis.write_data() does not write new topology information, but does remove topology information for removed atoms')
        print('Writing remaining velocity and topology sections...')
        end_idx = idx + n + 1
        for line in tqdm(lines[end_idx:]):
            header.append(line)
        
        out.writelines(header)


# class to save setting information
class Settings:
    def __init__(self):
        pass


def get_charge(frag, charges_dict):
    return round(np.sum([charges_dict[frag][atom]['charge'] for atom in charges_dict[frag]]),8)


# CPK color scheme
cpk_colors = {
    'H': '#FFFFFF', 'C': '#000000', 'N': '#3050F8', 'O': '#FF0D0D',
    'F': '#90E050', 'P': '#FF8000', 'S': '#FFFF30', 'Cl': '#1FF01F',
    'Br': '#A62929', 'I': '#940094', 'Na': '#AB5CF2', 'Mg': '#8AFF00',
    'Ca': '#3DFF00', 'Fe': '#E06633', 'Zn': '#7D80B0'
}

def build_graph_from_atoms(atom_group : mda.core.groups.AtomGroup, color_scheme : dict = cpk_colors) -> nx.Graph:
    '''
    Build a NetworkX graph from an MDAnalysis AtomGroup. Each atom is a node, and edges are bonds.

    Access the atoms in the graph via G.nodes[index]['atom'], where index is the atom index from the AtomGroup.

    Parameters
    ----------
    atom_group: MDAnalysis AtomGroup
        The group of atoms to build the graph from.
    color_scheme: dict, optional
        A dictionary mapping element symbols to colors for visualization. Defaults to CPK color scheme.
    '''

    G = nx.Graph()

    # add nodes to the graph
    for atom in atom_group:
        color = color_scheme.get(atom.element, '#808080')  # Default to gray if element not found
        G.add_node(atom.index, atom=atom, color=color, element=atom.element)

    # add edges to the graph, weights are the distances between atoms
    for bond in atom_group.bonds:
        atom1 = bond.atoms[0]
        atom2 = bond.atoms[1]
        distance = distances.distance_array(atom1.position, atom2.position)[0, 0]
        G.add_edge(atom1.index, atom2.index, weight=distance)
    
    return G


def solve_for_OH_position(O_position, C_position, CA_position, theta, d, h_dist=0.96):
    """
    Solve for one of the two possible positions of OH atom given the surrounding atoms O, C, and CA 
    and the force field angle and distance paramters

    OH     O
      \   //
        C
        |
        CA

    O_position, C_position, CA_position: 3D points as (3,) numpy arrays
    theta: angle at C (degrees)
    d: desired distance C-OH
    theta: angle between O-C-O (degrees)

    Returns: C1, H1 (one possible solution for C and H)
    """

    # save as numpy arrays
    theta = np.radians(theta)
    A = np.asarray(O_position, dtype=float)
    B = np.asarray(C_position, dtype=float)
    D = np.asarray(CA_position, dtype=float)
    
    # Unit vector BA
    BA = A - B
    BA = BA / np.linalg.norm(BA)
    
    # Compute plane normal from points A, B, D
    normal = np.cross(A - B, D - B)
    normal = normal / np.linalg.norm(normal)
    
    # Perpendicular direction in the plane (perp to BA but in ABD plane)
    perp = np.cross(normal, BA)
    perp = perp / np.linalg.norm(perp)
    
    # Rotate BA by ±theta in the plane spanned by (BA, perp)
    C1_dir = BA * np.cos(theta) + perp * np.sin(theta)
    # C2_dir = BA * np.cos(theta) - perp * np.sin(theta)
    
    # Scale to distance d and translate from B
    C1 = B + d * C1_dir
    # C2 = B + d * C2_dir

    def h_for_o(o_pos):
        vec_to_A = A - o_pos
        sign = np.dot(normal, vec_to_A)
        if sign > 0:
            h_dir = -normal
        else:
            h_dir = normal
        return o_pos + h_dist * h_dir

    H1 = h_for_o(C1)
    # H2 = h_for_o(C2)    
    
    return C1, H1


def solve_for_Cl_position(N_position, H_position,  d):
    """
    Solve for the position of Cl atom given the surrounding atoms N, H
    and the force field distance paramters -- place it along the N-H vector in the opposite direction

    H - N - Cl
        |
        C

    N_position, H_position: 3D points as (3,) numpy arrays
    d: desired distance N-Cl
    
    Returns: C
    """

    # save as numpy arrays
    A = np.asarray(H_position, dtype=float)
    B = np.asarray(N_position, dtype=float)
    
    # Unit vector BA
    BA = A - B
    BA = BA / np.linalg.norm(BA)
    
    # Scale to distance d and translate from B
    C = B + d * BA

    return C


if __name__ == '__main__':

    data = PolymAnalysis('prehydrate.data')

    mapping = {
        1 : 'c',
        2 : 'ca',
        3 : 'ha',
        4 : 'hn',
        5 : 'ho',
        6 : 'cl',
        7 : 'n',
        8 : 'nh',
        9 : 'o',
        10 : 'oh',
        11 : 'OW',
        12 : 'HW'
    }
    
    data._atomtype_mapping(mapping)
    data.write_GROMACS(output='tmp')
