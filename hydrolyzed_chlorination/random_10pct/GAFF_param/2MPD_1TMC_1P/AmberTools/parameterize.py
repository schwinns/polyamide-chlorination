#!/usr/bin/env python

import os
import subprocess
import argparse
import parmed as pmd

import numpy as np


def initialize():

    parser = argparse.ArgumentParser(description='Parameterize a solute with a force field')
    parser.add_argument('-i', '--input_file', type=str, help='Input coordinate file')
    parser.add_argument('-o', '--output_name', type=str, help='String for names of output files.')
    parser.add_argument('-m', '--mdp_file', type=str, help='gromacs .mdp file for energy minimization.')
    parser.add_argument('-e', '--espaloma', action='store_true', help='use espaloma_charge to assign charges')
    parser.add_argument('-n', '--net_charge', default=0, type=float, help='net charge on molecule')
    parser.add_argument('-ff', '--force_field', default='gaff', type=str, help='Name of force field used to assign'
                                                                               'parameters.')
    parser.add_argument('-f', '--format', default='gromacs', type=str, help='Format of output topology and coordinate'
                                                                            'files.')
    parser.add_argument('--tinker', action='store_true', help='Convert gromacs topolgy to Tinker .prm and determine '
                                                              'energy differences')
    parser.add_argument('--molcharge', action='store_true', help='Correct the partial charges from antechamber with am1bccsym from molcharge')

    return parser


class Parameterize:

    def __init__(self, input_file, output, mdp='em.gro', espaloma=False, output_format='gromacs', net_charge=0):
        """ Parameterize a molecule. Currently only GAFF is implemented.

        :param input_file: name of structure file in mol2 format
        :param output: name output residue
        :param mdp: name of mdp file with which to perform energy minimization
        :param espaloma: use espaloma_charge to assign partial charges
        :param output_format: Name of program for which output topology and coordinate files should be generated
        :param net_charge: net charge on molecule

        :type input_file: str
        :type output: str
        :type mdp: str
        :type espaloma: bool
        :type output_format: str
        :type net_charge: int
        """

        self.coord = input_file
        self.output = output
        self.mdp = mdp
        self.net_charge = net_charge
        self.espaloma = espaloma

        self.path = os.path.dirname(os.path.abspath(__file__))

        # A dictionary of file extensions for each type of implemented simulation software. The key is the name of the
        # simulation software in lower case. The value is a dictionary with two keys: 'coordinates' and 'topology'. The
        # value for each of these keys is the appropriate file extension for that type of file.
        self.extensions = {'gromacs': {'coordinates':'.gro', 'topology':'.top'}}
        self.format = output_format
        self.top = None
        self.mol2 = None

    def _antechamber(self):
        """ Run antechamber on input structure """

        if self.espaloma:
            # self._espaloma_charge()
            antechamber = 'antechamber -i %s -fi mol2 -o %s -fo mol2 -c rc -cf %s -s 2 -nc %d' % (self.coord,
                        self.output + '.mol2', self.output + '.crg', self.net_charge)
        else:
            antechamber = 'antechamber -i %s -fi mol2 -o %s -fo mol2 -c bcc -s 2 -nc %d' % (self.coord,
            self.output + '.mol2', self.net_charge)

        p = subprocess.Popen(antechamber.split())
        p.wait()

    def _espaloma_charge(self):
        """ Run espaloma_charge on input structure """

        from rdkit import Chem
        from espaloma_charge import charge

        print('Generating partial charges using EpsalomaCharge')
        molecule = Chem.MolFromMol2File(self.coord, removeHs=False)
        self.charges = charge(molecule)
        self._write_crg()

        # espaloma_charge = 'espaloma_charge -i %s -o %s.crg' % (self.coord, self.output) # NOT IMPLEMENTED
        # p = subprocess.Popen(espaloma_charge.split())
        # p.wait()

    def _write_crg(self):
        """ Write a crg file of partial charges for antechamber from numpy array """

        out = " " + np.array2string(self.charges, precision=6, max_line_width=81, floatmode="fixed")[1:-1]
        with open(self.output + '.crg', "w") as out_file:
            out_file.write(out)

    def _extract_charge(self):
        """ Get charges from mol2 file
        :param mol2: the input mol2 file in list format
        """

        with open(self.coord, 'r') as f:
            mol2 = []
            for line in f:
                mol2.append(line)

        # find the line where the atoms start being listed
        atoms = 0
        while mol2[atoms].count('ATOM') == 0:
            atoms += 1

        charges = []
        atoms += 1
        while mol2[atoms].count('@') == 0:
            charges.append(float(mol2[atoms][-10:(len(mol2[atoms]) - 1)]))
            atoms += 1

        charge = np.array(charges)

        # the total charge on the system needs to be corrected to the normalized charge
        correction = (float(self.net_charge) - sum(charge)) / len(charge)

        for i in range(len(charge)):
            charge[i] += correction

        # Correcting any small changes in formal charge (assumes rounding of 6 decimal places
        charge = np.around(charge, 6)
        if sum(charge) != float(self.net_charge):
            residual = sum(charge) - float(self.net_charge)
            placement = np.random.randint(len(charge), size=1)
            charge[placement] -= residual

        return charge

    def _parmchk(self):
        """ Run parmchk on GAFF parameters to check for missing parameters """

        parmchk = 'parmchk -i %s -f mol2 -o %s' % (self.output + '.mol2', self.output + '.frcmod')

        try:
            p = subprocess.Popen(parmchk.split())
        except FileNotFoundError:
            parmchk = 'parmchk2 -i %s -f mol2 -o %s' % (self.output + '.mol2', self.output + '.frcmod')
            p = subprocess.Popen(parmchk.split())

        p.wait()

    def _replace_charge(self, newcharges):
        """ Replace the charges in topology with new charges from molcharge
        """

        with open(self.top, 'r') as f:
            top = []
            for line in f:
                top.append(line)

        # find the line where the [ atoms ] section begins
        atoms = 0
        while top[atoms].count('[ atoms ]') == 0:
            atoms += 1

        atoms += 1
        count = 0
        s = '  '
        while count < len(newcharges):
            if top[atoms][0] == ";":
                atoms += 1
            else:
                a = top[atoms].split()
                a[6] = str(np.around(newcharges[count], 6))
                top[atoms] = s + a[0] + s + a[1] + s + a[2] + s + a[3] + s + a[4] + s + a[5] + s + a[6] + s + a[
                    7] + '\n'
                count += 1
                atoms += 1

        return top

    def _tleap(self):
        """ Write input to tleap then run tleap to generate AMBER topology files """

        tleap_in = "source leaprc.gaff\n" \
                   "SUS = loadmol2 " + self.output + ".mol2\n" \
                   "loadamberparams " + self.output + ".frcmod\n" \
                   "saveoff SUS sus.lib\n" \
                   "saveamberparm SUS " + self.output + ".prmtop " + self.output + ".inpcrd\n" \
                   "quit"

        with open('tleap.in', 'w') as f:
            f.write(tleap_in)

        tleap = 'tleap -f tleap.in'
        p = subprocess.Popen(tleap.split())
        p.wait()

    def assign_gaff_parameters(self):
        """ Call antechamber in order to assign atomtypes and charges
        """

        self._antechamber()
        self._parmchk()
        self._tleap()

    def assign_molcharge_charges(self):
        """ Use molcharge to assign charges based on a coordinate file
        """

        molcharge = "molcharge -in %s -out %s -method am1bccsym" % (self.coord, self.output + '.mol2')
        p = subprocess.Popen(molcharge.split())
        p.wait()

        self.mol2 = self.output + '.mol2'

    def convert_parameter_format(self):
        """ Use parmed to convert amber parameters to different format
        """

        self.coord = self.output + self.extensions[self.format.lower()]['coordinates']
        self.top = self.output + self.extensions[self.format.lower()]['topology']

        amber_params = pmd.load_file(self.output + ".prmtop", self.output + ".inpcrd")
        amber_params.save(self.top, overwrite=True)
        amber_params.save(self.coord, overwrite=True)

    def energy_minimize(self, convert_to_pdb=True):
        """ Energy minimize the

        :return:
        """

        from LLC_Membranes.llclib import gromacs
        from LLC_Membranes.setup.genmdp import SimulationMdp

        if not os.path.isfile('%s/%s' % (self.path, self.mdp)):
            mdp = SimulationMdp(self.coord)  # leave defaults
            mdp.write_em_mdp(self.mdp.split('.')[0])

        gromacs.simulate(self.mdp, self.top, self.coord, 'em', verbose=True)
        self.coord = 'em.gro'

        if convert_to_pdb:
            self.make_box(out='em.pdb')

    def insert_mol2_charges(self):

        updated_charges = self._extract_charge()
        new_top = self._replace_charge(updated_charges)

        with open(self.top, 'w') as f:
            for line in new_top:
                f.write(line)

    def make_box(self, d=3, out='box.gro'):
        """ Put a box around the solute

        :param d: minimum distance between solute and edge of box
        :param out: name of output boxed solute

        :type d: float
        :type out: str
        """

        from LLC_Membranes.llclib import gromacs

        gromacs.editconf(self.coord, out, d=d, center=True, box_type='cubic')
        self.coord = out


if __name__ == "__main__":

    args = initialize().parse_args()

    param = Parameterize(args.input_file, args.output_name, args.mdp_file, args.espaloma, output_format=args.format,
                         net_charge=args.net_charge)

    param.assign_gaff_parameters()
    param.convert_parameter_format()

