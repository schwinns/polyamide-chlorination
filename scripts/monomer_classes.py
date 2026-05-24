# Classes for MPD and TMC monomers, can be used for fragments within polymer chain

import MDAnalysis as mda
import numpy as np
    
class MPD():
    def __init__(self, N, xlink_n='7', ar_c='2', term_n='8', ar_h='3', hn='4'):
        '''
        N should be one of the 2 nitrogens on the MPD monomer
        
        Positions on MPD ring
               _1_
              4    2
              |    |
         N2 - 5_  _3 - N1
                6

        '''

        # Save input nitrogen and molecule information
        if N.type == xlink_n: 
            self.N1 = N # N1 is always crosslinked
            self.atoms = [self.N1]
        else:
            self.N2 = N
            self.atoms = [self.N2]
        self.mol = N.residue

        # Define the aromatic carbons by diagram above
        self.ca3 = [atom for atom in N.bonded_atoms if atom.type == ar_c][0]
        self.atoms.append(self.ca3)
        
        ca26 = [atom for atom in self.ca3.bonded_atoms if atom.type == ar_c]
        for ca in ca26:
            ca5 = [atom for atom in ca.bonded_atoms if atom != self.ca3 and (xlink_n in atom.bonded_atoms.types or term_n in atom.bonded_atoms.types)]
            if len(ca5) > 0:
                self.ca5 = ca5[0]
                self.atoms.append(self.ca5)
                self.ca6 = ca
                self.atoms.append(self.ca6)
            else:
                self.ca1 = [atom for atom in ca.bonded_atoms if atom.type == ar_c and atom != self.ca3][0]
                self.atoms.append(self.ca1)
                self.ca2 = ca
                self.atoms.append(self.ca2)

        self.ca4 = [atom for atom in self.ca5.bonded_atoms if atom.type == ar_c and atom != self.ca6][0]
        self.atoms.append(self.ca4)

        # Save other nitrogen
        if N.type == xlink_n:
            self.N2 = [atom for atom in self.ca5.bonded_atoms if atom.type == xlink_n or atom.type == term_n][0]
            self.atoms.append(self.N2)
        else:
            self.N1 = [atom for atom in self.ca5.bonded_atoms if atom.type == xlink_n or atom.type == term_n][0]
            self.atoms.append(self.N1)

        # Save aromatic hydrogens
        self.ha1 = [atom for atom in self.ca1.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha1)
        self.ha2 = [atom for atom in self.ca2.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha2)
        self.ha4 = [atom for atom in self.ca4.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha4)
        self.ha6 = [atom for atom in self.ca6.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha6)
    
        # Save hydrogens on nitrogens
        hn1 = [atom for atom in self.N1.bonded_atoms if atom.type == hn]
        self.hn1a = hn1[0]
        self.atoms.append(self.hn1a)
        if len(hn1) == 2:
            self.hn1b = hn1[1]
            self.atoms.append(self.hn1b)

        hn2 = [atom for atom in self.N2.bonded_atoms if atom.type == hn]
        self.hn2a = hn2[0]
        self.atoms.append(self.hn2a)
        if len(hn2) == 2:
            self.hn2b = hn2[1]
            self.atoms.append(self.hn2b)

        # Classify fragment as monomer, linear, or terminated
        n_hn = len(hn1) + len(hn2)
        if n_hn == 4:
            self.frag_type = 'mono'
            self.name = 'MPD'
        elif n_hn == 3:
            self.frag_type = 'term'
            self.name = 'MPD_T'
        elif n_hn == 2:
            self.frag_type = 'lin'
            self.name = 'MPD_L'
        else:
            raise AttributeError(f'cannot have number of hn type = {n_hn}')

    
    def assign_charges(self, charges):
        '''Assign charges to the fragment from the charges dictionary'''

        for a_type in charges.keys():
            charge = round(charges[a_type]['charge'], 8)
            atom = getattr(self, a_type)
            atom.charge = charge


class MPD_Cl_T():
    def __init__(self, N, cl='6', xlink_n='7', ar_c='2', term_n='8', ar_h='3', hn='4'):
        '''
        N should be one of the 2 nitrogens on the MPD monomer
        
        Positions on MPD ring
               _1_
         Cl   4    2
         |    |    |
         N2 - 5_  _3 - N1
                6

        '''

        # Save input nitrogen and molecule information
        if N.type == xlink_n: 
            self.N1 = N
            self.atoms = [self.N1]
        else:
            self.N2 = N
            self.atoms = [self.N2]
        self.mol = N.residue

        # Define the aromatic carbons by diagram above
        self.ca3 = [atom for atom in N.bonded_atoms if atom.type == ar_c][0]
        self.atoms.append(self.ca3)
        
        ca26 = [atom for atom in self.ca3.bonded_atoms if atom.type == ar_c]
        for ca in ca26:
            ca5 = [atom for atom in ca.bonded_atoms if atom != self.ca3 and (xlink_n in atom.bonded_atoms.types or term_n in atom.bonded_atoms.types)]
            if len(ca5) > 0:
                self.ca5 = ca5[0]
                self.atoms.append(self.ca5)
                self.ca6 = ca
                self.atoms.append(self.ca6)
            else:
                self.ca1 = [atom for atom in ca.bonded_atoms if atom.type == ar_c and atom != self.ca3][0]
                self.atoms.append(self.ca1)
                self.ca2 = ca
                self.atoms.append(self.ca2)

        self.ca4 = [atom for atom in self.ca5.bonded_atoms if atom.type == ar_c and atom != self.ca6][0]
        self.atoms.append(self.ca4)

        # Save other nitrogen
        if N.type == xlink_n:
            self.N2 = [atom for atom in self.ca5.bonded_atoms if atom.type == xlink_n or atom.type == term_n][0]
            self.atoms.append(self.N2)
        else:
            self.N1 = [atom for atom in self.ca5.bonded_atoms if atom.type == xlink_n or atom.type == term_n][0]
            self.atoms.append(self.N1)

        # Save aromatic hydrogens
        self.ha1 = [atom for atom in self.ca1.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha1)
        self.ha2 = [atom for atom in self.ca2.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha2)
        self.ha4 = [atom for atom in self.ca4.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha4)
        self.ha6 = [atom for atom in self.ca6.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha6)
    
        # Save hydrogens on nitrogens
        hn1 = [atom for atom in self.N1.bonded_atoms if atom.type == hn]
        self.hn1a = hn1[0]
        self.atoms.append(self.hn1a)
        if len(hn1) == 2:
            self.hn1b = hn1[1]
            self.atoms.append(self.hn1b)

        hn2 = [atom for atom in self.N2.bonded_atoms if atom.type == hn]
        self.hn2a = hn2[0]
        self.atoms.append(self.hn2a)
        if len(hn2) == 2:
            self.hn2b = hn2[1]
            self.atoms.append(self.hn2b)

        # save chlorines
        if len(hn1) == 1 and len(hn2) == 1 and self.N1.type == term_n and self.N2.type == term_n: # if N1 and N2 are terminal, and they both only have 1 H, they both must be chlorinated
            self.cl1 = [atom for atom in self.N1.bonded_atoms if atom.type == cl][0]
            self.cl2 = [atom for atom in self.N2.bonded_atoms if atom.type == cl][0]
        elif len(hn1) == 1 and self.N1.type == term_n: # if N1 is terminal and only has 1 H, it must be chlorinated
            self.cl = [atom for atom in self.N1.bonded_atoms if atom.type == cl][0]
        elif len(hn2) == 1 and self.N2.type == term_n: # if N2 is terminal and only has 1 H, it must be chlorinated
            self.cl = [atom for atom in self.N2.bonded_atoms if atom.type == cl][0]
        else:
            raise AttributeError(f'cannot find chlorine atom(s) on MPD_Cl_T fragment with nitrogen atom {N}')

        # Classify fragment
        n_hn = len(hn1) + len(hn2)
        if n_hn == 3: # for this class (MPD_Cl_T), the hn=3 case is MPD_Cl, that is a monomer with a Cl
            self.frag_type = 'mono'
            if 'cl' in self.N1.bonded_atoms.types: # just means that Cl is on N1
                self.name = 'MPD_Cl_1'
            elif 'cl' in self.N2.bonded_atoms.types:
                self.name = 'MPD_Cl_2'
            else:
                raise AttributeError(f'cannot find chlorine atom on MPD_Cl_T fragment with nitrogen atom {N}')
        elif n_hn == 2: # for this class (MPD_Cl_T), the hn=2 case can be MPD_T_Cl or a doubly chlorinated monomer
            if 'cl' in self.N1.bonded_atoms.types and 'cl' in self.N2.bonded_atoms.types:
                self.frag_type = 'mono'
                self.name = 'MPD_2Cl'
            else:
                self.frag_type = 'term'
                if self.cl in self.N1.bonded_atoms: # just means that Cl is on N1
                    self.name = 'MPD_T_Cl_1'
                elif self.cl in self.N2.bonded_atoms:
                    self.name = 'MPD_T_Cl_2'
                else:
                    raise AttributeError(f'cannot find chlorine atom on MPD_T_Cl fragment with nitrogen atom {N}')
        else:
            raise AttributeError(f'cannot have number of hn type = {n_hn}')

    
    def assign_charges(self, charges):
        '''Assign charges to the fragment from the charges dictionary'''

        for a_type in charges.keys():
            charge = round(charges[a_type]['charge'], 8)
            atom = getattr(self, a_type)
            atom.charge = charge        


class MPD_Cl_ortho():
    def __init__(self, N, cl='6', xlink_n='7', ar_c='2', term_n='8', ar_h='3', hn='4'):
        '''
        N should be one of the 2 nitrogens on the MPD monomer
        
        Positions on MPD ring
               _1_
              4    2
              |    |
         N2 - 5_  _3 - N1
                6
                |
                Cl

        '''

        # Save input nitrogen and molecule information
        if N.type == xlink_n: 
            self.N1 = N # N1 is always crosslinked
            self.atoms = [self.N1]
        else:
            self.N2 = N
            self.atoms = [self.N2]
        self.mol = N.residue

        # Define the aromatic carbons by diagram above
        self.ca3 = [atom for atom in N.bonded_atoms if atom.type == ar_c][0]
        self.atoms.append(self.ca3)
        
        ca26 = [atom for atom in self.ca3.bonded_atoms if atom.type == ar_c]
        for ca in ca26:
            ca5 = [atom for atom in ca.bonded_atoms if atom != self.ca3 and (xlink_n in atom.bonded_atoms.types or term_n in atom.bonded_atoms.types)]
            if len(ca5) > 0:
                self.ca5 = ca5[0]
                self.atoms.append(self.ca5)
                self.ca6 = ca
                self.atoms.append(self.ca6)
            else:
                self.ca1 = [atom for atom in ca.bonded_atoms if atom.type == ar_c and atom != self.ca3][0]
                self.atoms.append(self.ca1)
                self.ca2 = ca
                self.atoms.append(self.ca2)

        self.ca4 = [atom for atom in self.ca5.bonded_atoms if atom.type == ar_c and atom != self.ca6][0]
        self.atoms.append(self.ca4)

        # Save other nitrogen
        if N.type == xlink_n:
            self.N2 = [atom for atom in self.ca5.bonded_atoms if atom.type == xlink_n or atom.type == term_n][0]
            self.atoms.append(self.N2)
        else:
            self.N1 = [atom for atom in self.ca5.bonded_atoms if atom.type == xlink_n or atom.type == term_n][0]
            self.atoms.append(self.N1)

        # Save aromatic hydrogens
        self.ha1 = [atom for atom in self.ca1.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha1)
        self.ha2 = [atom for atom in self.ca2.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha2)
        self.ha4 = [atom for atom in self.ca4.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha4)

        # save either the aromatic hydrogen or chlorine on ca6
        is_chlorinated = len([atom for atom in self.ca6.bonded_atoms if atom.type == cl]) == 1
        if is_chlorinated:
            self.cl = [atom for atom in self.ca6.bonded_atoms if atom.type == cl][0]
            self.atoms.append(self.cl)
        else:
            self.ha6 = [atom for atom in self.ca6.bonded_atoms if atom.type == ar_h][0]
            self.atoms.append(self.ha6)
    
        # Save hydrogens on nitrogens
        hn1 = [atom for atom in self.N1.bonded_atoms if atom.type == hn]
        self.hn1a = hn1[0]
        self.atoms.append(self.hn1a)
        if len(hn1) == 2:
            self.hn1b = hn1[1]
            self.atoms.append(self.hn1b)

        hn2 = [atom for atom in self.N2.bonded_atoms if atom.type == hn]
        self.hn2a = hn2[0]
        self.atoms.append(self.hn2a)
        if len(hn2) == 2:
            self.hn2b = hn2[1]
            self.atoms.append(self.hn2b)

        # Classify fragment as monomer, linear, or terminated
        n_hn = len(hn1) + len(hn2)
        if n_hn == 4:
            self.frag_type = 'mono'
            self.name = 'MPD'
        elif n_hn == 3:
            self.frag_type = 'term'
            self.name = 'MPD_T'
        elif n_hn == 2:
            self.frag_type = 'lin'
            self.name = 'MPD_L'
        else:
            raise AttributeError(f'cannot have number of hn type = {n_hn}')
        
        if is_chlorinated:
            self.name += '_Cl_ortho'

    
    def assign_charges(self, charges):
        '''Assign charges to the fragment from the charges dictionary'''

        for a_type in charges.keys():
            charge = round(charges[a_type]['charge'], 8)
            atom = getattr(self, a_type)
            atom.charge = charge


class MPD_Cl_para():
    def __init__(self, N, cl='6', xlink_n='7', ar_c='2', term_n='8', ar_h='3', hn='4'):
        '''
        N should be one of the 2 nitrogens on the MPD monomer
        
        Positions on MPD ring
               _1_
              4    2 - Cl
              |    |
         N2 - 5_  _3 - N1
                6

        '''

        # Save input nitrogen and molecule information
        near_cl = len([dih for dih in N.dihedrals if 'cl' in dih.atoms.types]) > 0
        if near_cl: 
            self.N1 = N # N1 is always near Cl
            self.atoms = [self.N1]
        else:
            self.N2 = N
            self.atoms = [self.N2]
        self.mol = N.residue

        # Define the aromatic carbons by diagram above
        if near_cl:
            self.ca3 = [atom for atom in N.bonded_atoms if atom.type == ar_c][0]
            self.atoms.append(self.ca3)
            
            ca26 = [atom for atom in self.ca3.bonded_atoms if atom.type == ar_c]
            for ca in ca26:
                ca5 = [atom for atom in ca.bonded_atoms if atom != self.ca3 and (xlink_n in atom.bonded_atoms.types or term_n in atom.bonded_atoms.types)]
                if len(ca5) > 0:
                    self.ca5 = ca5[0]
                    self.atoms.append(self.ca5)
                    self.ca6 = ca
                    self.atoms.append(self.ca6)
                else:
                    self.ca1 = [atom for atom in ca.bonded_atoms if atom.type == ar_c and atom != self.ca3][0]
                    self.atoms.append(self.ca1)
                    self.ca2 = ca
                    self.atoms.append(self.ca2)

            self.ca4 = [atom for atom in self.ca5.bonded_atoms if atom.type == ar_c and atom != self.ca6][0]
            self.atoms.append(self.ca4)

        else: # if my N is N2, not near Cl
            self.ca5 = [atom for atom in N.bonded_atoms if atom.type == ar_c][0]
            self.atoms.append(self.ca5)

            ca46 = [atom for atom in self.ca5.bonded_atoms if atom.type == ar_c]
            for ca in ca46:
                ca3 = [atom for atom in ca.bonded_atoms if atom != self.ca5 and (xlink_n in atom.bonded_atoms.types or term_n in atom.bonded_atoms.types)]
                if len(ca3) > 0:
                    self.ca3 = ca3[0]
                    self.atoms.append(self.ca3)
                    self.ca6 = ca
                    self.atoms.append(self.ca6)
                else:
                    self.ca1 = [atom for atom in ca.bonded_atoms if atom.type == ar_c and atom != self.ca5][0]
                    self.atoms.append(self.ca1)
                    self.ca4 = ca
                    self.atoms.append(self.ca4)

            self.ca2 = [atom for atom in self.ca3.bonded_atoms if atom.type == ar_c and atom != self.ca6][0]
            self.atoms.append(self.ca2)

        # Save other nitrogen
        if near_cl:
            self.N2 = [atom for atom in self.ca5.bonded_atoms if atom.type == xlink_n or atom.type == term_n][0]
            self.atoms.append(self.N2)
        else:
            self.N1 = [atom for atom in self.ca3.bonded_atoms if atom.type == xlink_n or atom.type == term_n][0]
            self.atoms.append(self.N1)

        # Save aromatic hydrogens
        self.ha1 = [atom for atom in self.ca1.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha1)
        self.ha4 = [atom for atom in self.ca4.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha4)
        self.ha6 = [atom for atom in self.ca6.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha6)

        # save either the aromatic hydrogen or chlorine on ca2
        is_chlorinated = len([atom for atom in self.ca2.bonded_atoms if atom.type == cl]) == 1
        if is_chlorinated:
            self.cl = [atom for atom in self.ca2.bonded_atoms if atom.type == cl][0]
            self.atoms.append(self.cl)
        else:
            self.ha2 = [atom for atom in self.ca2.bonded_atoms if atom.type == ar_h][0]
            self.atoms.append(self.ha2)
    
        # Save hydrogens on nitrogens
        hn1 = [atom for atom in self.N1.bonded_atoms if atom.type == hn]
        self.hn1a = hn1[0]
        self.atoms.append(self.hn1a)
        if len(hn1) == 2:
            self.hn1b = hn1[1]
            self.atoms.append(self.hn1b)

        hn2 = [atom for atom in self.N2.bonded_atoms if atom.type == hn]
        self.hn2a = hn2[0]
        self.atoms.append(self.hn2a)
        if len(hn2) == 2:
            self.hn2b = hn2[1]
            self.atoms.append(self.hn2b)

        # Classify fragment as monomer, linear, or terminated
        n_hn = len(hn1) + len(hn2)
        if n_hn == 4:
            self.frag_type = 'mono'
            self.name = 'MPD'
        elif n_hn == 3:
            self.frag_type = 'term'
            self.name = 'MPD_T'
        elif n_hn == 2:
            self.frag_type = 'lin'
            self.name = 'MPD_L'
        else:
            raise AttributeError(f'cannot have number of hn type = {n_hn}')
        
        if is_chlorinated:
            self.name += '_Cl_para'

    
    def assign_charges(self, charges):
        '''Assign charges to the fragment from the charges dictionary'''

        for a_type in charges.keys():
            charge = round(charges[a_type]['charge'], 8)
            atom = getattr(self, a_type)
            atom.charge = charge    


class TMC():
    def __init__(self, C, ar_c='2', xlink_c='1', ar_h='3', deprot_o='9', prot_o='10', ho_type='5'):
        '''
        C should be one of the 3 carbonyl carbons on TMC
        
        Positions on TMC ring
            C1
            |
           _1_
          4    2
          |    |
        C3-5_  _3-C2
            6
                
        '''

        # Save input carbon and molecule information
        self.C1 = C
        self.mol = C.residue
        self.atoms = [self.C1]

        # Define the aromatic carbons by diagram above
        self.ca1 = [atom for atom in C.bonded_atoms if atom.type == ar_c][0]
        self.atoms.append(self.ca1)
        self.ca2 = [atom for atom in self.ca1.bonded_atoms if atom.type == ar_c][0]
        self.atoms.append(self.ca2)
        self.ca3 = [atom for atom in self.ca2.bonded_atoms if atom.type == ar_c and atom != self.ca1][0]
        self.atoms.append(self.ca3)
        self.ca6 = [atom for atom in self.ca3.bonded_atoms if atom.type == ar_c and atom != self.ca2][0]
        self.atoms.append(self.ca6)
        self.ca5 = [atom for atom in self.ca6.bonded_atoms if atom.type == ar_c and atom != self.ca3][0]
        self.atoms.append(self.ca5)
        self.ca4 = [atom for atom in self.ca5.bonded_atoms if atom.type == ar_c and atom != self.ca6][0]
        self.atoms.append(self.ca4)

        # Save other carbonyl carbons
        self.C2 = [atom for atom in self.ca3.bonded_atoms if atom.type == xlink_c][0]
        self.atoms.append(self.C2)
        self.C3 = [atom for atom in self.ca5.bonded_atoms if atom.type == xlink_c][0]
        self.atoms.append(self.C3)

        # Save aromatic hydrogens
        self.ha2 = [atom for atom in self.ca2.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha2)
        self.ha4 = [atom for atom in self.ca4.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha4)
        self.ha6 = [atom for atom in self.ca6.bonded_atoms if atom.type == ar_h][0]
        self.atoms.append(self.ha6)

        # Save carbonyl oxygens
        self.O1 = [atom for atom in self.C1.bonded_atoms if atom.type == deprot_o][0]
        self.atoms.append(self.O1)
        self.O2 = [atom for atom in self.C2.bonded_atoms if atom.type == deprot_o][0]
        self.atoms.append(self.O2)
        self.O3 = [atom for atom in self.C3.bonded_atoms if atom.type == deprot_o][0]
        self.atoms.append(self.O3)

        # Classify fragment as monomer, terminated, linear, or crosslinked and save hydroxide groups
        n_oh = 0
        n_dp = 0
        c_types = []
        for i,c in enumerate([self.C1, self.C2, self.C3]):
            oh = [atom for atom in c.bonded_atoms if atom.type == prot_o]
            o = [atom for atom in c.bonded_atoms if atom.type == deprot_o]
            if len(oh) > 0:
                c_types.append('term_P')
                n_oh += 1
                setattr(self, f'oh{i+1}', oh[0])
                self.atoms.append(getattr(self, f'oh{i+1}'))
                ho = [atom for atom in oh[0].bonded_atoms if atom.type == ho_type]
                setattr(self, f'ho{i+1}', ho[0])
                self.atoms.append(getattr(self, f'ho{i+1}'))
            elif len(o) == 2:
                dp_o = [atom for atom in o if atom not in self.atoms][0]
                c_types.append('term_D')
                n_dp += 1
                setattr(self, f'O{n_dp+3}', dp_o)
                self.atoms.append(getattr(self, f'O{n_dp+3}'))
            else:
                c_types.append('xlink')
    
        if n_oh == 0 and n_dp == 0:
            self.frag_type = 'xlink'
            self.name = 'TMC_C'
        elif n_oh == 0 and n_dp == 1:
            self.frag_type = 'lin'
            if c_types[0] == 'term_D':
                self.name = 'TMC_L_D_1'
            elif c_types[1] == 'term_D':
                self.name = 'TMC_L_D_2'
            elif c_types[2] == 'term_D':
                self.name = 'TMC_L_D_3'
        elif n_oh == 0 and n_dp == 2:
            self.frag_type = 'term'
            if c_types[0] == 'xlink':
                self.name = 'TMC_T_0P_1'
            elif c_types[1] == 'xlink':
                self.name = 'TMC_T_0P_2'
            elif c_types[2] == 'xlink':
                self.name = 'TMC_T_0P_3'
        elif n_oh == 1 and n_dp == 0:
            self.frag_type = 'lin'
            if c_types[0] == 'term_P':
                self.name = 'TMC_L_P_1'
            elif c_types[1] == 'term_P':
                self.name = 'TMC_L_P_2'
            elif c_types[2] == 'term_P':
                self.name = 'TMC_L_P_3'
        elif n_oh == 1 and n_dp == 1:
            self.frag_type = 'term'
            if c_types[0] == 'xlink' and c_types[1] == 'term_P':
                self.name = 'TMC_T_1P_1_2'
            elif c_types[0] == 'xlink' and c_types[2] == 'term_P':
                self.name = 'TMC_T_1P_1_3'
            elif c_types[1] == 'xlink' and c_types[0] == 'term_P':
                self.name = 'TMC_T_1P_2_1'
            elif c_types[1] == 'xlink' and c_types[2] == 'term_P':
                self.name = 'TMC_T_1P_2_3'
            elif c_types[2] == 'xlink' and c_types[0] == 'term_P':
                self.name = 'TMC_T_1P_3_1'
            elif c_types[2] == 'xlink' and c_types[1] == 'term_P':
                self.name = 'TMC_T_1P_3_2'
        elif n_oh == 2 and n_dp == 0:
            self.frag_type = 'term'
            if c_types[0]== 'xlink':
                self.name = 'TMC_T_2P_1'
            elif c_types[1] == 'xlink':
                self.name = 'TMC_T_2P_2'
            elif c_types[2] == 'xlink':
                self.name = 'TMC_T_2P_3'
        elif n_oh == 3:
            self.frag_type = 'mono'
            self.name = 'TMC_3P'
        elif n_oh == 1 and n_dp == 2:
            self.frag_type = 'mono'
            if c_types[0] == 'term_P': # this just means that C1 is protonated
                self.name = 'TMC_1P_1'
            elif c_types[1] == 'term_P': # C2 is protonated
                self.name = 'TMC_1P_2'
            elif c_types[2] == 'term_P': # C3 is protonated
                self.name = 'TMC_1P_3'
        elif n_oh == 2 and n_dp == 1:
            self.frag_type = 'mono'
            if c_types[0] == 'term_D': # C1 is deprotonated
                self.name = 'TMC_2P_1'
            elif c_types[1] == 'term_D': # C2 is deprotonated
                self.name = 'TMC_2P_2'
            elif c_types[2] == 'term_D': # C3 is deproton
                self.name = 'TMC_2P_3'
        elif n_dp == 3:
            self.frag_type = 'mono'
            self.name = 'TMC_0P'
        else:
            raise AttributeError(f'cannot have number of protonated Os = {n_oh} and number of deprotonated Os = {n_dp}')


    def assign_charges(self, charges):
        '''Assign charges to the fragment from the charges dictionary'''

        for a_type in charges.keys():
            charge = round(charges[a_type]['charge'], 8)
            atom = getattr(self, a_type)
            atom.charge = charge    
            


def check_N(N, nh_type='8', n_type='7', ar_c_type='2'):

    # input the atom N to check the other N
    #   if the other N is type NH, then MPD-T
    #   if the other N is type LN, then MPD-L

    # Positions on MPD ring
    #       _1_
    #     4    2
    #     |    |
    # N - 5_  _3 - N
    #       6

    if N.type == nh_type:
        n_NH = 1
    elif N.type == n_type:
        n_NH = 0
    else:
        raise ValueError(f'{N} is not a N')
    
    CA3 = [atom for atom in N.bonded_atoms if atom.type == ar_c_type][0]
    CA26 = [atom for atom in CA3.bonded_atoms if atom.type == ar_c_type]
    for CA in CA26:
        n_NH += len([atom for atom in CA.bonded_atoms if atom != CA3 and atom.type == ar_c_type and nh_type in atom.bonded_atoms.types])

    return n_NH


def check_C(C):

    # input the atom C to check the other Cs
    #   if n_CT = 0, then TMC-C
    #   if n_CT = 1, then TMC-L
    #   if n_CT = 2, then TMC-T

    # Positions on TMC ring
    #        C(14)
    #        |
    #       _1_
    #     4    2
    #     |    |
    # C - 5_  _3 - C
    #       6

    n_CT = len([atom for atom in C.bonded_atoms if atom.type == '10'])
    CA1 = [atom for atom in C.bonded_atoms if atom.type == '2'][0]
    CA2 = [atom for atom in CA1.bonded_atoms if atom.type == '2'][0]
    CA3 = [atom for atom in CA2.bonded_atoms if atom.type == '2' and atom != CA1][0]
    n_CT += len([atom for atom in CA3.bonded_atoms if atom.type == '1' and '10' in atom.bonded_atoms.types])
    CA6 = [atom for atom in CA3.bonded_atoms if atom.type == '2' and atom != CA2][0]
    CA5 = [atom for atom in CA6.bonded_atoms if atom.type == '2' and atom != CA3][0]
    n_CT += len([atom for atom in CA5.bonded_atoms if atom.type == '1' and '10' in atom.bonded_atoms.types])

    return n_CT

        
if __name__ == '__main__':

    u = mda.Universe('testing/2MPD-2TMC.mol2')

    atoms = u.atoms

    # for mol2, convert to numeric atom types
    atom_mapping = {
        'C.2' : '1',
        'C.ar' : '2',
        'H' : {
            'C.ar' : '3',
            'N.am' : '4',
            'N.pl3' : '4',
            'O.3' : '5'
        },
        'Cl' : '6',
        'N.am' : '7',
        'N.pl3' : '8',
        'O.2' : '9',
        'O.3' : '10'   
    }

    num_types = []
    for atom in atoms:
        if atom.type != 'H':
            num_types.append(atom_mapping[atom.type])
        else:
            bonded_type = atom.bonded_atoms[0].type
            num_types.append(atom_mapping[atom.type][bonded_type])

    atoms.types = num_types
    
    # Create MPD and TMC fragments
    mpds = []
    Ns = []
    for N in u.select_atoms('type 7'):
        mpd = MPD(N)
        if mpd.N2 not in Ns:
            mpds.append(mpd)
        
        Ns.append(mpd.N1)
        Ns.append(mpd.N2)

    tmcs = []
    Cs = []
    for C in u.select_atoms('type 1'):
        tmc = TMC(C)
        if tmc.C2 not in Cs and tmc.C3 not in Cs:
            tmcs.append(tmc)

        Cs.append(tmc.C1)
        Cs.append(tmc.C2)
        Cs.append(tmc.C3)


    # Assign charges to MPD and TMC fragments
    import yaml
    
    with open('charges.yaml', 'r') as file: # read in charges from yaml file
        charges_dict = yaml.safe_load(file)

    total_charge = 0
    for mono in mpds + tmcs:
        print(mono.name)
        mono.assign_charges(charges_dict[mono.name])
        frag_charge = np.array([a.charge for a in mono.atoms]).sum()
        total_charge += frag_charge


    print('New total charge in system: {:.4f}'.format(total_charge))
