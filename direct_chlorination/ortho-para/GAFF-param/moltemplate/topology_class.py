# Defines a Topology class from a provided .top file
# Currently does not account for #include statements

def remove_comments(line):
    return line.split(';')[0]

class Topology():

    def __init__(self, topology_file):

        f = open(topology_file, 'r')

        # HEADER INFORMATION #
        for line in f:
            # Go to defaults directive
            if line.startswith('[ defaults ]'):
                break

        # DEFAULTS DIRECTIVE #
        for line in f:
            # Go to atomtypes directive
            if line.startswith('[ atomtypes ]'):
                break
            
            # Ignore comments and blank lines
            elif not line.startswith(';') and len(line.split()) != 0:

                line = remove_comments(line)
                if line.split()[2] == 'yes':
                    gen_pairs = True
                else:
                    gen_pairs = False

                self.defaults_directive = {
                    'nbfunc'    : int(line.split()[0]),
                    'comb-rule' : int(line.split()[1]),
                    'gen-pairs' : gen_pairs,
                    'fudgeLJ'   : float(line.split()[3]),
                    'fudgeQQ'   : float(line.split()[4])
                }

        # ATOMTYPES DIRECTIVE #
        self.atomtypes = {}
        for line in f:
            # Go to moleculetype directive
            if line.startswith('[ moleculetype ]'):
                break

            # Ignore comments and blank lines
            elif not line.startswith(';') and len(line.split()) != 0:

                line = remove_comments(line)
                atom = line.split()
                self.atomtypes[atom[0]] = {
                    'at_num'  : int(atom[1]),
                    'mass'    : float(atom[2]),
                    'charge'  : float(atom[3]),
                    'ptype'   : atom[4],
                    'sigma'   : float(atom[5]),
                    'epsilon' : float(atom[6]) 
                }

        # MOLECULETYPE DIRECTIVE #
        self.moleculetype = {}
        for line in f:
            # Go to atoms directive
            if line.startswith('[ atoms ]'):
                break

            # Ignore comments and blank lines
            elif not line.startswith(';') and len(line.split()) != 0:

                line = remove_comments(line)
                self.moleculetype[line.split()[0]] = {
                    'nrexcl'  : int(line.split()[1])
                }

        # ATOMS DIRECTIVE #
        self.atoms = {}
        self.atoms_by_name = {}  # create an additional attribute where atoms are accessible by name in topology file
        for line in f:
            # Go to bonds directive
            if line.startswith('[ bonds ]'):
                break

            # Ignore comments and blank lines
            elif not line.startswith(';') and len(line.split()) != 0:

                line = remove_comments(line)
                atom = line.split()
                atom_id = int(atom[0])
                self.atoms[atom_id] = {
                    'atom_id'   : int(atom[0]),
                    'type'      : atom[1],
                    'resnr'     : int(atom[2]),
                    'residue'   : atom[3],
                    'atom_name' : atom[4],
                    'cgnr'      : int(atom[5]),
                    'charge'    : float(atom[6]),
                    'mass'      : float(atom[7])
                }

                atom_name = atom[4]
                self.atoms_by_name[atom_name] = self.atoms[atom_id]

        # BONDS DIRECTIVE #
        self.bonds = {}
        bond_id = 0
        for line in f:
            # Go to angles directive
            if line.startswith('[ pairs ]'):
                break

            # Ignore comments and blank lines
            elif not line.startswith(';') and len(line.split()) != 0:

                line = remove_comments(line)
                bond = line.split()
                bond_id += 1
                self.bonds[bond_id] = {
                    'a1'     : int(bond[0]),
                    'a2'     : int(bond[1]),
                    'func'   : int(bond[2]),
                    'params' : [float(p) for p in bond[3:]]
                }

        # PAIRS DIRECTIVE #
        self.pairs = {}
        pair_id = 0
        for line in f:
            # Go to angles directive
            if line.startswith('[ angles ]'):
                break

            # Ignore comments and blank lines
            elif not line.startswith(';') and len(line.split()) != 0:

                line = remove_comments(line)
                pair = line.split()
                pair_id += 1
                self.pairs[pair_id] = {
                    'a1'     : int(pair[0]),
                    'a2'     : int(pair[1]),
                    'func'   : int(pair[2]),
                    'params' : [float(p) for p in pair[3:]]
                }

        # ANGLES DIRECTIVE #
        self.angles = {}
        angle_id = 0
        for line in f:
            # Go to dihedrals directive
            if line.startswith('[ dihedrals ]'):
                break

            # Ignore comments and blank lines
            elif not line.startswith(';') and len(line.split()) != 0:

                line = remove_comments(line)
                angle = line.split()
                angle_id += 1
                self.angles[angle_id] = {
                    'a1'     : int(angle[0]),
                    'a2'     : int(angle[1]),
                    'a3'     : int(angle[2]),
                    'func'   : int(angle[3]),
                    'params' : [float(p) for p in angle[4:]]
                }

        # DIHEDRALS DIRECTIVE #
        self.dihedrals = {}
        dih_id = 0
        for line in f:
            # Go to system directive
            if line.startswith('[ system ]'):
                break

            # Ignore comments and blank lines
            elif not line.startswith(';') and len(line.split()) != 0:

                line = remove_comments(line)
                dih = line.split()
                dih_id += 1
                self.dihedrals[dih_id] = {
                    'a1'     : int(dih[0]),
                    'a2'     : int(dih[1]),
                    'a3'     : int(dih[2]),
                    'a4'     : int(dih[3]),
                    'func'   : int(dih[4]),
                    'params' : [float(p) for p in dih[5:]]
                }

        # SYSTEM DIRECTIVE #
        for line in f:
            # Go to molecules directive
            if line.startswith('[ molecules ]'):
                break

            # Ignore comments and blank lines
            elif not line.startswith(';') and len(line.split()) != 0:

                line = remove_comments(line)
                self.name = line.split('\n')[0]

        # MOLECULES DIRECTIVE #
        self.molecules = {}
        for line in f:

            # Ignore comments and blank lines
            if not line.startswith(';') and len(line.split()) != 0:

                line = remove_comments(line)
                self.molecules[line.split()[0]] = {
                    'nmol'  : int(line.split()[1])
                }