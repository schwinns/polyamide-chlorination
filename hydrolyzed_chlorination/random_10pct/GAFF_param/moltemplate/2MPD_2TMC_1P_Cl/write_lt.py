# Script to write moltemplate input files

# Imports
from topology_class import Topology
import mdtraj as md
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gro', help='gro file of the molecule to be templated')
parser.add_argument('-p', '--top', help='topology file of the molecule to be templated')
parser.add_argument('-o', '--output', default='output.lt', help='name of the output lt file')
args = parser.parse_args()

gro = args.gro
top = args.top
output = args.output

coordinate = md.load(gro) # mdtraj atoms are 0-indexed
topology = Topology(top) # Topology atoms are 1-indexed
coords = coordinate.xyz[0,:,:]

# Write the name of Moltemplate object
mol = output.split('.')[0]
out = open(output, 'w')

out.write('import "gaff.lt"\n\n')
out.write(mol + ' inherits GAFF {\n\n')

# Data Atoms section
out.write('\twrite("Data Atoms") {\n')
out.write('\t\t# atomID\tmolID\tatomType\tcharge\tx\ty\tz\n')

i = 0
coord_atoms = []
for atom in coordinate.top.atoms:

    top_atom = topology.atoms_by_name[atom.name]
    coord_atoms.append(atom.name)

    atomID = top_atom['atom_name']
    molID = top_atom['residue']
    atomType = top_atom['type']
    charge = top_atom['charge']
    x = coords[i,0]*10
    y = coords[i,1]*10
    z = coords[i,2]*10

    line = '\t\t$atom:{:<2}\t$mol:{}\t@atom:{:<2}\t{}\t{}\t{}\t{}\n'.format(atomID, molID, atomType, charge, x, y, z)
    out.write(line)
    i += 1

out.write('\t}\n\n')

# Data Bond List section
out.write('\twrite("Data Bond List") {\n')
out.write('\t\t# bondID\tbondType\tatomID1\tatomID2\n')

for i in topology.bonds:

    bond = topology.bonds[i]

    a1 = bond['a1']
    a2 = bond['a2']

    a1_name = topology.atoms[a1]['atom_name']
    a2_name = topology.atoms[a2]['atom_name']

    if a1_name not in coord_atoms:
        print('{} not in coordinate file, not writing bond {} ({}-{}) to {}'.format(a1_name, i, a1_name, a2_name, output))

    elif a2_name not in coord_atoms:
        print('{} not in coordinate file, not writing bond {} ({}-{}) to {}'.format(a2_name, i, a1_name, a2_name, output))
    
    else:
        bond_name = a1_name + a2_name

        line = '\t\t$bond:{:<4}\t$atom:{:<2}\t$atom:{:<2}\n'.format(bond_name, a1_name, a2_name)
        out.write(line)

out.write('\t}\n\n')
out.write('} # ' + mol)
out.close()
