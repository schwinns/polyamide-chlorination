# Classes for reading a Gromacs topology file

from collections import Counter
from copy import copy
import numpy as np

class AtomType:
    '''Class to store atom type information'''
    def __init__(self, line):

        l = line.split()
        self.type = l[0]
        self.bondingtype = l[1]
        self.atomic_number = int(l[2])
        self.mass = float(l[3])
        self.charge = float(l[4])
        self.ptype = l[5]
        self.sigma = float(l[6])   # nm
        self.epsilon = float(l[7]) # kJ/mol


    def __repr__(self):
        return f'<AtomType {self.type}>'
    

    def __str__(self):
        return f'{self.type}'

    
    def copy(self):
        return copy(self)


class Bond:
    '''Class to store bond information'''
    def __init__(self, a1, a2, btype, params):

        self.atoms = [a1, a2]
        self.type = btype
        self._params = params
        
        # parse parameters based on btype
        # bond functional forms per Gromacs documentation
        if btype == 1: # bond (i.e. harmonic)
            self.b0 = params[0] # nm
            self.kb = params[1] # kJ/mol/nm^2
        elif btype == 2: # G96 bond
            self.b0 = params[0] # nm
            self.kb = params[1] # kJ/mol/nm^4
        elif btype == 3: # Morse
            self.b0 = params[0] # nm
            self.D = params[1] # kJ/mol
            self.beta = params[2] # nm^-1
        elif btype == 4: # cubic bond
            self.b0 = params[0] # nm
            self.C2 = params[1] # kJ/mol/nm^2
            self.C3 = params[2] # kJ/mol/nm^3
        elif btype == 5: # connection
            pass # no parameters
        elif btype == 6: # harmonic potential
            self.b0 = params[0] # nm
            self.kb = params[1] # kJ/mol/nm^2
        elif btype == 7: # FENE bond
            self.bm = params[0] # nm
            self.kb = params[1] # kJ/mol/nm^2
        else:
            raise TypeError(f'Bond type {btype} not implemented')


    def __repr__(self):
        return f'<Bond of type {self.type} between {self.atoms[0]} and {self.atoms[1]}>'
    

class Angle:
    '''Class to store angle information'''
    def __init__(self, a1, a2, a3, angtype, params):

        self.atoms = [a1, a2, a3]
        self.type = angtype
        self._params = params
        
        # parse parameters based on angtype
        # angle functional forms per Gromacs documentation
        if angtype == 1: # angle (i.e. harmonic)
            self.theta0 = params[0] # deg
            self.ktheta = params[1] # kJ/mol/rad^2
        elif angtype == 2: # G96 angle
            self.theta0 = params[0] # deg
            self.ktheta = params[1] # kJ/mol
        elif angtype == 3: # cross bond-bond
            self.r1e = params[0] # nm
            self.r2e = params[1] # nm
            self.krr = params[2] # kJ/mol/nm^2
        elif angtype == 4: # cross bond-angle
            self.r1e = params[0] # nm
            self.r2e = params[1] # nm
            self.r3e = params[2] # nm
            self.krtheta = params[3] # kJ/mol/nm^2
        elif angtype == 5: # Urey-Bradley
            self.theta0 = params[0] # deg
            self.ktheta = params[1] # kJ/mol/rad^2
            self.r13 = params[2] # nm
            self.kUB = params[3] # kJ/mol/nm^2
        elif angtype == 6: # quartic angle
            self.theta0 = params[0] # deg
            self.C0 = params[1] # kJ/mol
            self.C1 = params[2] # kJ/mol/rad
            self.C2 = params[3] # kJ/mol/rad^2
            self.C3 = params[4] # kJ/mol/rad^3
            self.C4 = params[5] # kJ/mol/rad^4
        else:
            raise TypeError(f'Angle type {angtype} not implemented')


    def __repr__(self):
        return f'<Angle of type {self.type} between {self.atoms[0]}, {self.atoms[1]}, and {self.atoms[2]}>'


class Dihedral:
    '''Class to store angle information'''
    def __init__(self, a1, a2, a3, a4, dihtype, params):

        self.atoms = [a1, a2, a3, a4]
        self.type = dihtype
        self._params = params
        
        # parse parameters based on dihtype
        # dihedral functional forms per Gromacs documentation
        if dihtype == 1: # proper dihedral
            self.phis = params[0] # deg
            self.kphi = params[1] # kJ/mol
            self.n = params[2] # multiplicity
        elif dihtype == 2: # improper dihedral
            self.ksee0 = params[0] # deg
            self.kksee = params[1] # kJ/mol/rad^2
        elif dihtype == 3: # Ryckaert-Bellemans dihedral
            self.C0 = params[0] # kJ/mol
            self.C1 = params[1] # kJ/mol
            self.C2 = params[2] # kJ/mol
            self.C3 = params[3] # kJ/mol
            self.C4 = params[4] # kJ/mol
            self.C5 = params[5] # kJ/mol
        elif dihtype == 4: # periodic improper dihedral
            self.phis = params[0] # deg
            self.kphi = params[1] # kJ/mol
            self.n = params[2] # multiplicity
        elif dihtype == 5: # Fourier dihedral
            self.C1 = params[0] # kJ/mol
            self.C2 = params[1] # kJ/mol
            self.C3 = params[2] # kJ/mol
            self.C4 = params[3] # kJ/mol
        elif dihtype == 9: # proper dihedral (multiple)
            self.phis = params[0] # deg
            self.kphi = params[1] # kJ/mol
            self.n = params[2] # multiplicity
        elif dihtype == 10: # restricted dihedral
            self.phi0 = params[0] # deg
            self.kphi = params[1] # kJ/mol
        elif dihtype == 11: # combined bending-torsion potential
            self.kphi = params[0] # kJ/mol
            self.a0 = params[1]
            self.a1 = params[2]
            self.a2 = params[3]
            self.a3 = params[4]
            self.a4 = params[5]
        else:
            raise TypeError(f'Dihedral type {dihtype} not implemented')


    def __repr__(self):
        return f'<Dihedral of type {self.type} between {self.atoms[0]}, {self.atoms[1]}, {self.atoms[2]}, and {self.atoms[3]}>'


class MolObj:
    '''Generic class to store information about other topology directives'''
    def __init__(self, atoms, ftype, params):

        self.atoms = atoms
        self.type = ftype
        self._params = params

    def __repr__(self):
        rep = f'<MolObj of type {self.type} between {self.atoms[0]}'
        for atom in self.atoms[1:]:
            rep += f', {atom}'
        rep += '>'
        return rep

    
class Atom:
    '''Class for an individual atom in the system'''
    def __init__(self, idx, line):
        
        # save the atom index within the whole topology
        self.idx = idx

        # save all topology information
        l = line.split()
        self.num = int(l[0])
        self.type = l[1]
        self.resnum = int(l[2])
        self.resname = l[3]
        self.atomname = l[4]
        self.cgnr = int(l[5])
        self.charge = float(l[6])
        self.mass = float(l[7])

        # create lists for bonds, angles, dihedrals, etc.
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.settles = []

        # create an attribute for the AtomType and Molecule
        self.atomtype = None
        self.molecule = None


    def __repr__(self):
        return f'<Atom {self.idx} of type {self.atomtype} in Molecule {self.molecule}>'


    def __str__(self):
        return f'Atom {self.idx}'


class Molecule:
    '''Class for an individual molecule in the system'''
    def __init__(self, name, number):

        self.name = name
        self.n_mols = number
        self.nex = 3 # default value for number of bonds away to exclude non-bonded interactions
        self.directives = []

        
    def __repr__(self):
        return f'<Molecule {self.name} with {self.n_atoms} atoms>'


    def __str__(self):
        return f'{self.name}'


    @property
    def n_atoms(self):
        return len(self.atoms)  

    
class GromacsTopology:
    def __init__(self, top_file, verbose=False):
    
        self.gmx = 'gmx' # name of Gromacs executable
        self.verbose = verbose
    
        # save information about original file
        self.filename = top_file
        self.original = open(top_file, 'r').readlines()
        tmp = [line for line in self.original if not line.startswith(';')]
        self.cleaned = [line for line in tmp if not len(line.split()) == 0]

        # read included files and add to self.cleaned
        incl_idx = [i for i,line in enumerate(self.cleaned) if line.startswith('#include')]

        if len(incl_idx) > 0:
            incl_files = []
            for idx in incl_idx:
                line = self.cleaned[idx]
                incl_file = line.split()[1].strip('"')
                incl = open(incl_file, 'r').readlines()
                tmp = [line for line in incl if not line.startswith(';')]
                incl_clean = [line for line in tmp if not len(line.split()) == 0]
                incl_files.append(incl_clean)

            print(f'WARNING: Assumes #include statements are on consecutive lines with indices: {incl_idx}. If these indices are not consecutive, please adjust GromacsTopology.init().')
            cleaned = self.cleaned[:incl_idx[0]]

            for i in range(len(incl_idx)):
                cleaned += incl_files[i]

            cleaned += self.cleaned[incl_idx[-1]+1:]
            self.cleaned = cleaned

        # initialize some objects to store data
        self.n_atoms = 0
        self.atoms = []
        self.molecules = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []

        # read in directives
        self.read_defaults()
        atomtypes_end_idx = self.read_atomtypes()
        self.read_molecules()
    
        # read in each molecule
        for mol in self.molecules:
            idx = [i+atomtypes_end_idx for i,line in enumerate(self.cleaned[atomtypes_end_idx:]) if line.startswith(mol.name)][0]
            self.read_moleculetype(mol, idx)

        # hard-code reading system directive at end of file
        if not hasattr(self, 'system'):
            idx = self.cleaned.index('[ system ]\n')
            self._parse_system(self.cleaned[idx+1:idx+2])

        # convert lists into numpy arrays to access with indices
        self.atoms = np.array(self.atoms)
        self.molecules = np.array(self.molecules)
        self.bonds = np.array(self.bonds)
        self.angles = np.array(self.angles)
        self.dihedrals = np.array(self.dihedrals)

    
    def _parse_atoms(self, molecule, lines):
        '''Parse atoms and add to topology, molecule'''

        setattr(molecule, 'atoms', [])

        for line in lines:
            self.n_atoms += 1
            atom = Atom(self.n_atoms - 1, line)
            atom.atomtype = self.atomtypes[atom.type]
            atom.molecule = molecule
            self.atoms.append(atom)
            molecule.atoms.append(atom)

        if self.verbose:
            print(f'\nMolecule {molecule} has {len(molecule.atoms)} atoms')
            

    def _parse_bonds(self, molecule, lines):
        '''Parse bonds and add to topology, atoms, molecule'''

        setattr(molecule, 'bonds', [])
        molecule.directives.append('bonds')
        
        for line in lines:
            l = line.split()
            
            a1 = int(l[0])
            a2 = int(l[1])
            btype = int(l[2])
            params = [float(p) for p in l[3:]]

            atom1 = molecule.atoms[a1-1]
            atom2 = molecule.atoms[a2-1]

            bond = Bond(atom1, atom2, btype, params)
            self.bonds.append(bond)
            molecule.bonds.append(bond)
            atom1.bonds.append(bond)
            atom2.bonds.append(bond)

        if self.verbose:
            print(f'Molecule {molecule} has {len(molecule.bonds)} bonds')

    
    def _parse_angles(self, molecule, lines):
        '''Parse angles and add to topology, atoms, molecule'''
        
        setattr(molecule, 'angles', [])
        molecule.directives.append('angles')
        
        for line in lines:
            l = line.split()
            
            a1 = int(l[0])
            a2 = int(l[1])
            a3 = int(l[2])
            angtype = int(l[3])
            params = [float(p) for p in l[4:]]

            atom1 = molecule.atoms[a1-1]
            atom2 = molecule.atoms[a2-1]
            atom3 = molecule.atoms[a3-1]

            angle = Angle(atom1, atom2, atom3, angtype, params)
            self.angles.append(angle)
            molecule.angles.append(angle)
            atom1.angles.append(angle)
            atom2.angles.append(angle)
            atom3.angles.append(angle)
            
        if self.verbose:
            print(f'Molecule {molecule} has {len(molecule.angles)} angles')
        

    def _parse_dihedrals(self, molecule, lines):
        '''Parse dihedrals and add to topology, atoms, molecule'''
        
        setattr(molecule, 'dihedrals', [])
        molecule.directives.append('dihedrals')
        
        for line in lines:
            l = line.split()
            
            a1 = int(l[0])
            a2 = int(l[1])
            a3 = int(l[2])
            a4 = int(l[3])
            dihtype = int(l[4])
            params = [float(p) for p in l[5:]]

            atom1 = molecule.atoms[a1-1]
            atom2 = molecule.atoms[a2-1]
            atom3 = molecule.atoms[a3-1]
            atom4 = molecule.atoms[a4-1]

            dihedral = Dihedral(atom1, atom2, atom3, atom4, dihtype, params)
            self.dihedrals.append(dihedral)
            molecule.dihedrals.append(dihedral)
            atom1.dihedrals.append(dihedral)
            atom2.dihedrals.append(dihedral)
            atom3.dihedrals.append(dihedral)
            atom4.dihedrals.append(dihedral)
            
        if self.verbose:
            print(f'Molecule {molecule} has {len(molecule.dihedrals)} dihedrals')

    
    def _parse_settles(self, molecule, lines):
        '''Parse settles and add to atoms'''

        setattr(molecule, 'settles', [])
        molecule.directives.append('settles')
    
        for line in lines:
            l = line.split()

            atom = molecule.atoms[int(l[0])-1]
            ftype = int(l[1])
            params = [float(d) for d in l[2:]]

            settle = MolObj([atom], ftype, params)

            atom.settles = params
            molecule.settles.append(settle)
        
        if self.verbose:
            print(f'Molecule {molecule} has {len(molecule.settles)} SETTLES')


    def _parse_exclusions(self, molecule, lines):
        '''Parse exlcusions and add to atoms, molecule'''

        setattr(molecule, 'exclusions', [])
        molecule.directives.append('exclusions')

        for line in lines:
            atoms = [molecule.atoms[int(i)-1] for i in line.split()]
            atom = atoms[0]
            
            params = []
            ftype = ''

            exclusion = MolObj(atoms, ftype, params)

            atom.exclusions = atoms[1:]
            molecule.exclusions.append(exclusion)

        
        if self.verbose:
            print(f'Molecule {molecule} has {len(molecule.exclusions)} exclusions')
    

    def _parse_system(self, lines):
        '''Parse system directive'''
        
        self.system = lines[0].split('\n')[0]
    
        if self.verbose:
            print(f'\nFinished reading topology of system: {self.system}')


    def read_defaults(self):
        '''Read [ defaults ] directive from top file'''
    
        idx = self.cleaned.index('[ defaults ]\n')

        line = self.cleaned[idx+1].split()
        self.nbfunc = int(line[0])
        self.comb_rule = int(line[1])
        self.gen_pairs = line[2]
        self.fudgeLJ = float(line[3])
        self.fudgeQQ = float(line[4])
    
    
    def read_atomtypes(self):
        '''Read [ atomtypes ] directive from top file'''

        print('\nWARNING: only reads atom types with sigma and epsilon parameters\n')
    
        idx = self.cleaned.index('[ atomtypes ]\n')
    
        self.atomtypes = {}
        for i,line in enumerate(self.cleaned[idx+1:]):
            if line.startswith('['):
                atomtypes_end_idx = i
                break
            else:
                atype = AtomType(line)
                self.atomtypes[atype.type] = atype

        if self.verbose:
            print(f'{len(self.atomtypes)} atom types in the topology')

        return atomtypes_end_idx


    def read_molecules(self):
        '''Read [ molecules ] directive from top file'''

        idx = self.cleaned.index('[ molecules ]\n')

        self.molecules = []
        n_mols = 0
        for line in self.cleaned[idx+1:]:
            if line.startswith('['):
                break
            else:
                mol = Molecule(line.split()[0], int(line.split()[1]))
                self.molecules.append(mol)
                n_mols += int(line.split()[1])
    
        print(f'{len(self.molecules)} molecule types in the topology for a total of {n_mols} molecules')


    def read_moleculetype(self, molecule, idx):
        '''Read all information for a given moleculetype (i.e. atoms, bonds, angles, dihedrals, etc.)'''

        # get topology lines for molecule
        idx_end = [i+idx for i,line in enumerate(self.cleaned[idx:]) if line.startswith('[ molecule')][0]
        my_lines = self.cleaned[idx+1:idx_end]

        # find directive header lines (e.g. '[ atoms ]', '[ bonds ]', ...)
        dir_idx = [i for i,line in enumerate(my_lines) if line.strip().startswith('[')]
        if len(dir_idx) == 0:
            return

        # sentinel -- end of last directive is end of my_lines
        dir_idx.append(len(my_lines))

        # iterate directives and parse their block (start+1 : next_start)
        for i in range(len(dir_idx)-1):
            start = dir_idx[i]
            end = dir_idx[i+1]
            header = my_lines[start].strip()
            block = my_lines[start+1:end]

            if header.startswith('[ atoms ]'):
                self._parse_atoms(molecule, block)
            elif header.startswith('[ bonds ]'):
                self._parse_bonds(molecule, block)
            elif header.startswith('[ settles ]'):
                self._parse_settles(molecule, block)
            elif header.startswith('[ angles ]'):
                self._parse_angles(molecule, block)
            elif header.startswith('[ dihedrals ]'):
                # block now correctly contains all dihedral lines up to next directive / end
                self._parse_dihedrals(molecule, block)
            elif header.startswith('[ exclusions ]'):
                self._parse_exclusions(molecule, block)
            elif header.startswith('[ system ]'):
                self._parse_system(block)
            else:
                if '[' in header:
                    dir_type = header.split('[ ')[1].split(' ]')[0]
                    raise TypeError(f'Cannot parse directive type [ {dir_type} ]')


    def write(self, filename):
        '''Write the (possibly modified) topology'''

        all_lines = []

        # write [ defaults ] directive
        directive = ['[ defaults ]\n']
        directive.append(f'\t{self.nbfunc}\t{self.comb_rule}\t{self.gen_pairs}\t{self.fudgeLJ}\t{self.fudgeQQ}\n')
        directive.append('\n')
        all_lines.extend(directive)

        # write [ atomtypes ] directive
        directive = ['[ atomtypes ]\n']
        for a in self.atomtypes:
            at = self.atomtypes[a]
            directive.append(f'{at.type:4s} {at.bondingtype:4s}\t{at.atomic_number:.0f}\t{at.mass:.8f}\t{at.charge:.8f}\tA\t{at.sigma:.8e}\t{at.epsilon:.8e}\n')

        directive.append('\n')
        all_lines.extend(directive)

        # loop through all molecules
        for molecule in self.molecules:

            # do not duplicate moleculetype if already written
            if f'{molecule.name}\t\t{molecule.nex}\n' in all_lines:
                continue

            # write [ moleculetype ] directive
            directives = ['[ moleculetype ]\n']
            directives.append(f'{molecule.name}\t\t{molecule.nex}\n')
            directives.append('\n')

            # write [ atoms ] directive
            directives.append('[ atoms ]\n')
            for atom in molecule.atoms:
                directives.append(f'{atom.num:7d} {atom.type:4s}\t\t{atom.resnum} {atom.resname:8s} {atom.atomname:9s}\t{atom.cgnr:d}\t{atom.charge:.8f}\t{atom.mass:.8f}\n')

            directives.append('\n')

            # write other molecule directives
            for directive in molecule.directives:
                directives.append(f'[ {directive} ]\n')
                for mol_obj in getattr(molecule, directive):
                    line = ''
                    for atom in mol_obj.atoms:
                        line += f'{atom.num:>7d}\t'
                    
                    line += f'{mol_obj.type}\t'
                    for param in mol_obj._params:
                        line += f'{param:.8e}\t'

                    line += '\n'
                    directives.append(line)

                directives.append('\n')

            all_lines.extend(directives)

        # write [ system ] directive
        directive = ['[ system ]\n']
        directive.append(f'{self.system}\n')
        directive.append('\n')
        all_lines.extend(directive)

        # write [ molecules ] directive
        directive = ['[ molecules ]\n']
        for molecule in self.molecules:
            directive.append(f'{molecule.name}\t{molecule.n_mols}\n')

        all_lines.extend(directive)

        with open(filename, 'w') as out:
            out.writelines(all_lines)

        print(f'Finished writing topology to {filename}!')

    def get_bond(self, atomtype1, atomtype2):
        '''Get the bond between two atom types in a topology.'''

        atomtypes = [atomtype1, atomtype2]

        for b in self.bonds:
            my_types = [a.type for a in b.atoms]
            if Counter(my_types) == Counter(atomtypes):
                return b
        return None


    def get_angle(self, atomtype1, atomtype2, atomtype3):
        '''Get the angle among three atom types in a topology.'''

        atomtypes = [atomtype1, atomtype2, atomtype3]

        for a in self.angles:
            my_types = [a.type for a in a.atoms]
            if Counter(my_types) == Counter(atomtypes):
                return a
        return None


    def get_dihedral(self, atomtype1, atomtype2, atomtype3, atomtype4, is_improper=False):
        '''Get the dihedral among four atom types in a topology. Should specify if this dihedral is improper.'''

        atomtypes = [atomtype1, atomtype2, atomtype3, atomtype4]
        improper_types = [2, 4] # either harmonic (2) or periodic (4) improper

        for d in self.dihedrals:
            my_types = [a.type for a in d.atoms]
            if Counter(my_types) == Counter(atomtypes):
                if is_improper and d.type in improper_types:
                    return d
                
                if not is_improper and d.type not in improper_types:
                    return d
                
        return None
