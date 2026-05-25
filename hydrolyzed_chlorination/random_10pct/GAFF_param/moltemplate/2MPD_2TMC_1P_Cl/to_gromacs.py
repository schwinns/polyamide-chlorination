# Script to convert prehydration LAMMPS files to Gromacs

from polym_analysis import PolymAnalysis

data = PolymAnalysis('cleanedsystem.data')

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
    10 : 'oh'
}

data._atomtype_mapping(mapping)
data.write_GROMACS(output='2MPD_2TMC_1P_Cl_converted')