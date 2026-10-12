# Run all analysis for chlorinated membrane
# hydrogen bonding, RDFs, swelling, density profiles

from glob import glob
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

import MDAnalysis as mda
import MDAnalysis.transformations as trans
from ParallelMDAnalysis import ParallelInterRDF as InterRDF
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from diffusion_coefficient import DiffusionCoefficient
# from solvation_analysis.solute import Solute
from monomer_classes import MPD, MPD_Cl_T, TMC, MPD_Cl_para


def save_object(obj, filename):
    '''Save object to pickle file'''
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def create_MPD_fragments(universe):
    '''Create MPD fragment objects'''

    mpds = []
    Ns = []
    for N in tqdm(universe.atoms.select_atoms('type n nh')):

        has_cl_on_N = 'cl' in N.bonded_atoms.types

        if not has_cl_on_N: # check other N
            my_ca = [atom for atom in N.bonded_atoms if atom.type == 'ca'][0] # get connected aromatic C
            next_ca = [atom for atom in my_ca.bonded_atoms if atom.type == 'ca'] # get next aromatic C
            for ca in next_ca:

                next_next_ca = [atom for atom in ca.bonded_atoms if atom != my_ca and atom.type == 'ca'][0]
                if 'N' in next_next_ca.bonded_atoms.elements:
                    n = [atom for atom in next_next_ca.bonded_atoms if atom.element == 'N'][0]
                    break

            has_cl_on_N = 'cl' in n.bonded_atoms.types

        # check for cl on aromatic ring
        my_ca = [atom for atom in N.bonded_atoms if atom.type == 'ca'][0] # get connected aromatic C
        ortho_cas = [atom for atom in my_ca.bonded_atoms if atom.type == 'ca'] # get ortho aromatic Cs
        for ca in ortho_cas:
            if 'cl' in ca.bonded_atoms.types:
                has_cl_on_ca = True
                break
            else:
                has_cl_on_ca = False

        if not has_cl_on_ca:
            meta_cas = [atom for atom in ortho_cas for atom in atom.bonded_atoms if atom.type == 'ca' and atom != my_ca] # get meta aromatic Cs
            para_ca = [atom for atom in meta_cas for atom in atom.bonded_atoms if atom.type == 'ca' and atom not in ortho_cas] # get para aromatic Cs
            para_ca = list(dict.fromkeys(para_ca))[0] # Preserve traversal order while removing duplicates from para aromatic carbons.
            has_cl_on_ca = 'cl' in para_ca.bonded_atoms.types

        if has_cl_on_N:
            mpd = MPD_Cl_T(N, cl='cl', xlink_n='n', ar_c='ca', term_n='nh', ar_h='ha', hn='hn')
        elif has_cl_on_ca:
            mpd = MPD_Cl_para(N, cl='cl', xlink_n='n', ar_c='ca', term_n='nh', ar_h='ha', hn='hn')
        else:
            mpd = MPD(N, xlink_n='n', ar_c='ca', term_n='nh', ar_h='ha', hn='hn')

        if mpd.N2 not in Ns:
            mpds.append(mpd)
        
        Ns.append(mpd.N1)
        Ns.append(mpd.N2)

    # convert to atom groups
    all_mpds = mda.AtomGroup([a for mpd in mpds for a in mpd.atoms])

    for mpd in mpds:
        mpd.atoms = mda.AtomGroup(mpd.atoms)

    return mpds, all_mpds


def create_TMC_fragments(universe):
    '''Create TMC fragment objects'''

    tmcs = []
    Cs = []
    for C in tqdm(u.atoms.select_atoms('type c')):
        tmc = TMC(C, ar_c='ca', xlink_c='c', ar_h='ha', deprot_o='o', prot_o='oh', ho_type='ho')
        if tmc.C2 not in Cs and tmc.C3 not in Cs:
            tmcs.append(tmc)

        Cs.append(tmc.C1)
        Cs.append(tmc.C2)
        Cs.append(tmc.C3)

    # convert to atom groups
    all_tmcs = mda.AtomGroup([a for tmc in tmcs for a in tmc.atoms])

    for tmc in tmcs:
        tmc.atoms = mda.AtomGroup(tmc.atoms)

    return tmcs, all_tmcs


def determine_membrane_bounds(universe, weighted='number', return_hist=False):
    '''Determine membrane bounds based on intersection of the polymer density and water density profiles'''

    if weighted == "number":
        water = universe.select_atoms('(resname SOL) and (not element H)')
        polymer = universe.select_atoms('(resname PA*) and (not element H)')
    elif weighted == "mass":
        water = universe.select_atoms('resname SOL')
        polymer = universe.select_atoms('resname PA*')

    # duplicate in z for consistent PBC
    water_pos = np.zeros(water.n_atoms*2)
    polymer_pos = np.zeros(polymer.n_atoms*2)

    max_dims = max([ts.dimensions[2] for ts in u.trajectory])
    bin_width = 1
    bins = np.arange(0, max_dims+bin_width, bin_width)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    water_hist = np.histogram([], bins=np.arange(0, max_dims+bin_width, bin_width))[0].astype(np.float64)
    polymer_hist = np.histogram([], bins=np.arange(0, max_dims+bin_width, bin_width))[0].astype(np.float64)
    nonzero_frames = [
        np.histogram([], bins=np.arange(0, max_dims+bin_width, bin_width))[0],
        np.histogram([], bins=np.arange(0, max_dims+bin_width, bin_width))[0]
    ]

    for ts in tqdm(universe.trajectory):
        water_pos[:water.n_atoms] = water.positions[:,2]
        water_pos[water.n_atoms:] = water.positions[:,2] + ts.dimensions[2]
        polymer_pos[:polymer.n_atoms] = polymer.positions[:,2]
        polymer_pos[polymer.n_atoms:] = polymer.positions[:,2] + ts.dimensions[2]

        # Recalculate histograms
        hist, b = np.histogram(water_pos, bins=np.arange(0, max_dims+bin_width, bin_width), weights=water.masses if weighted == 'mass' else None)
        hist = hist.astype(np.float64) / (bin_width * ts.dimensions[0] * ts.dimensions[1])
        water_hist += hist
        water_bin_centers = 0.5 * (b[:-1] + b[1:])

        hist, b = np.histogram(polymer_pos, bins=np.arange(0, max_dims+bin_width, bin_width), weights=polymer.masses if weighted == 'mass' else None)
        hist = hist.astype(np.float64) / (bin_width * ts.dimensions[0] * ts.dimensions[1])
        polymer_hist += hist

    # Average over frames
    water_hist /= universe.trajectory.n_frames
    polymer_hist /= universe.trajectory.n_frames

    # Find intersections by checking sign changes of the difference
    difference = water_hist - polymer_hist
    sign_changes = np.where(np.diff(np.sign(difference)))[0]

    print(f"Found {len(sign_changes)} intersection points:")
    intersections = []
    for idx in sign_changes:
        # Linear interpolation to find more precise intersection
        z1, z2 = water_bin_centers[idx], water_bin_centers[idx + 1]
        d1, d2 = difference[idx], difference[idx + 1]
        
        # Interpolate
        z_intersect = z1 - d1 * (z2 - z1) / (d2 - d1)
        density_intersect = np.interp(z_intersect, water_bin_centers, water_hist)
        
        intersections.append((z_intersect, density_intersect))
        print(f"  Z = {z_intersect:.2f} Å, Density = {density_intersect:.4f} amu/Å³")

    # Mark intersections
    if intersections:
        z_intersects = [pt[0] for pt in intersections]
        d_intersects = [pt[1] for pt in intersections]

    membrane_bounds = np.linspace(intersections[0][0], intersections[-1][0], 5) # split membrane into quarters
    print(f'Membrane bounds (Z): {membrane_bounds}')

    if return_hist:
        return membrane_bounds, water_hist, polymer_hist, bin_centers
    else:
        return membrane_bounds


def hydrogen_bonds(universe, donors_sel='(type n) or (type nh)', acceptors_sel='(type o)', hydrogens_sel='(type hn)'):
    '''Calculate the number of hydrogen bonds over time'''

    hbonds = HBA(universe=universe,
                donors_sel=donors_sel,
                acceptors_sel=acceptors_sel,
                hydrogens_sel=hydrogens_sel)

    hbonds.run(
        start=None,
        stop=None,
        step=None,
        verbose=True
    )

    return hbonds.count_by_time()


def calculate_volume(universe):
    '''Calculate the volume of the middle 50% of polymer over time'''

    polymer = universe.select_atoms('resname PA*')

    membrane_bounds = np.zeros((universe.trajectory.n_frames, 5))
    xy_area = np.zeros(universe.trajectory.n_frames)
    for i,ts in enumerate(tqdm(universe.trajectory)):

        polymer_min = np.nonzero(polymer.positions[:,2] < 10)[0]
        if len(polymer_min) > 0: # there are some extreme cases where the polymer jumps due to PBC
            # Create a mask to exclude indices in polymer_min
            mask = np.ones(polymer.positions.shape[0], dtype=bool)  # Start with all True
            mask[polymer_min] = False  # Set indices in polymer_min to False
            min_pos = polymer.positions[mask,2].min()

        else:
            min_pos = polymer.positions[:,2].min()

        membrane_bounds[i,:] = np.linspace(min_pos, polymer.positions[:,2].max(), num=5)
        xy_area[i] = (universe.dimensions[0]*universe.dimensions[1]) # A^2

    vol = xy_area * (membrane_bounds[:,3] - membrane_bounds[:,1])  # A^3 of the middle 50% of membrane
    return vol, xy_area, np.column_stack((membrane_bounds[:,1], membrane_bounds[:,3]))


def calculate_RDFs(universe, membrane_lower_bound=43, membrane_upper_bound=99, njobs=1):
    '''Calculate RDFs for water molecules in the bulk membrane'''

    waters_in_membrane = universe.select_atoms(f'type OW and prop z > {membrane_lower_bound} and prop z < {membrane_upper_bound}', updating=True)
    polymer = universe.select_atoms('resname PA*')
    cl = universe.select_atoms('type cl')
    cations = universe.select_atoms('resname NA')
    anions = universe.select_atoms('resname CL')
    cooh = universe.select_atoms('type oh')
    coo_c = universe.select_atoms('(type c) and (not bonded type n) and (not bonded type oh)')
    coo_o = universe.select_atoms('(type o) and (bonded group coo_c)', coo_c=coo_c)
    amide_n = universe.select_atoms('(type n) and (bonded type c)')
    amide_c = universe.select_atoms('(type c) and (bonded type n)')
    amide_o = universe.select_atoms('(type o) and (bonded group amide_c)', amide_c=amide_c)
    amine_n = universe.select_atoms('type nh')
    ca = universe.select_atoms('type ca')

    groups = [waters_in_membrane, polymer, polymer.select_atoms('not element H'), cl, cations, 
            anions, cooh, coo_c, coo_o, amide_n, amide_o, amine_n, ca]
    labels = ['Water', 'Polymer', 'Polymer (no H)', 'polymer Cl', 'Na+', 
            'Cl-', 'COOH', 'COO-C', 'COO-O', 'Amide N', 'Amide O', 'Amine N', 'aromatic']

    rdf_results = np.zeros((len(groups)+1, 150)) # 150 bins, 10 rdfs, and r values

    for i,g2 in enumerate(groups):
        print(f'\tCalculating RDF for {labels[i]}')
        irdf = InterRDF(waters_in_membrane, g2, nbins=150, range=(0.0, 15.0), verbose=True)
        irdf.run(njobs=njobs)
        rdf_results[i+1,:] = irdf.results.rdf

    rdf_results[0,:] = irdf.results.bins
    df = pd.DataFrame(rdf_results.T, columns=['r'] + labels)

    return df


def average_PSD(frames):
    '''Average pore size distributions from multiple frames'''

    psd_list = []
    for frame in frames:
        data = np.loadtxt(frame, skiprows=1)
        psd_list.append(data[:,1])

    psd_array = np.array(psd_list)
    bins = data[:,0]
    psd_mean = psd_array.mean(axis=0)
    psd_std = psd_array.std(axis=0)

    return bins, psd_mean, psd_std


def calculate_density(universe, bin_width=0.5):
    '''Calculate density profiles for different species and average density of bulk membrane'''

    # get density profiles and membrane bounds
    max_dims = max([ts.dimensions[2] for ts in universe.trajectory])
    bins = np.arange(0, max_dims+bin_width, bin_width)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    membrane_bounds, water_density_profile, polymer_density_profile, b = determine_membrane_bounds(universe, return_hist=True)

    # get MPD and TMC fragments
    mpds, all_mpds = create_MPD_fragments(universe)
    tmcs, all_tmcs = create_TMC_fragments(universe)

    # calculate bulk membrane density time series
    water_density = np.zeros((u.trajectory.n_frames, 4)) # time, xy, number, mass
    polymer_density = np.zeros((u.trajectory.n_frames, 11)) # time, xy, total number, total mass, MPD number, MPD mass, MPD frags, TMC number, TMC mass, TMC frags, C number

    for i,ts in tqdm(enumerate(u.trajectory)):
        bulk_membrane = u.select_atoms(f'prop z > {membrane_bounds[1]} and prop z < {membrane_bounds[3]}')
        # bulk_membrane = u.select_atoms(f'prop z > {membrane_bounds[0]} and prop z < {membrane_bounds[-1]}')
        my_water = bulk_membrane.select_atoms('resname SOL')
        my_polymer = bulk_membrane.select_atoms('resname PA*')
        my_MPD = all_mpds & my_polymer
        my_TMC = all_tmcs & my_polymer

        n_mpd = len([mpd for mpd in mpds if mpd.atoms.issubset(my_MPD)])
        n_tmc = len([tmc for tmc in tmcs if tmc.atoms.issubset(my_TMC)])

        xy = (ts.dimensions[0] * ts.dimensions[1]) # Angstroms^2

        water_density[i,0] = ts.time
        water_density[i,1] = xy
        water_density[i,2] = my_water.n_atoms # number
        water_density[i,3] = my_water.total_mass() # mass

        polymer_density[i,0] = ts.time
        polymer_density[i,1] = xy
        polymer_density[i,2] = my_polymer.n_atoms
        polymer_density[i,3] = my_polymer.total_mass()
        polymer_density[i,4] = my_MPD.n_atoms
        polymer_density[i,5] = my_MPD.total_mass()
        polymer_density[i,6] = n_mpd
        polymer_density[i,7] = my_TMC.n_atoms
        polymer_density[i,8] = my_TMC.total_mass()
        polymer_density[i,9] = n_tmc
        polymer_density[i,10] = len(my_polymer.select_atoms('element C'))


    df_water = pd.DataFrame(water_density, columns=['time', 'area', 'water_number', 'water_mass'])
    df_polymer = pd.DataFrame(polymer_density, columns=['time', 'area', 'polymer_number', 'polymer_mass', 'MPD_number', 'MPD_mass', 'MPD_frags', 'TMC_number', 'TMC_mass', 'TMC_frags', 'C_number'])
    density_df = pd.merge(df_water, df_polymer, on=['time', 'area'])

    return bin_centers, water_density_profile, polymer_density_profile, density_df


def coordination_analysis(universe):
    '''Get the coordination environment around water molecules in the membrane'''

    waters = universe.select_atoms('type OW') # only water oxygen atoms
    cations = universe.select_atoms('resname NA')
    anions = universe.select_atoms('resname CL')

    xlink_c = universe.select_atoms(f'(type c) and (bonded type n)')
    coo_c = universe.select_atoms(f'(type c) and (not bonded type oh n)')

    cooh_oh = universe.select_atoms(f'type oh')
    amide_o = universe.select_atoms(f'(type o) and (bonded group xlink_c)', xlink_c=xlink_c)
    amide_n = universe.select_atoms(f'(type n) and (bonded group xlink_c)', xlink_c=xlink_c)
    coo_o = universe.select_atoms(f'(type o) and (bonded group coo_c)', coo_c=coo_c)
    nh2 = universe.select_atoms(f'type nh')
    cl = universe.select_atoms('type cl')
    ca = universe.select_atoms('type ca')

    solute = Solute.from_atoms(waters,
                        {
                            'amide_o' : amide_o,
                            'cooh_oh' : cooh_oh,
                            'coo_o' : coo_o,
                            'nh2' : nh2,
                            'anions' : anions,
                            'cations' : cations,
                            'cl' : cl,
                            'water' : waters,
                            'aromatic' : ca,
                            'amide_n' : amide_n
                        },
                        solute_name='Water', radii={'nh2' : 4.0, 'anions' : 3.95, 'cl' : 5.1, 'aromatic' : 4.35})

    solute.run()

    return solute


def water_interactions(universe):
    '''Calculate the fraction of water molecules interacting with different functional groups'''

    waters = universe.select_atoms('type OW') # only water oxygen atoms
    cations = universe.select_atoms('resname NA')
    anions = universe.select_atoms('resname CL')

    xlink_c = universe.select_atoms(f'(type c) and (bonded type n)')
    coo_c = universe.select_atoms(f'(type c) and (not bonded type oh n)')

    cooh_oh = universe.select_atoms(f'type oh')
    amide_o = universe.select_atoms(f'(type o) and (bonded group xlink_c)', xlink_c=xlink_c)
    amide_n = universe.select_atoms(f'(type n) and (bonded group xlink_c)', xlink_c=xlink_c)
    coo_o = universe.select_atoms(f'(type o) and (bonded group coo_c)', coo_c=coo_c)
    nh2 = universe.select_atoms(f'type nh')
    cl = universe.select_atoms('type cl')
    ca = universe.select_atoms('type ca')

    groups = {
                'amide_o' : amide_o,
                'cooh_oh' : cooh_oh,
                'coo_o' : coo_o,
                'nh2' : nh2,
                'anions' : anions,
                'cations' : cations,
                'cl' : cl,
                'water' : waters,
                'aromatic' : ca,
                'amide_n' : amide_n
            }
    
    solutes = {}
    for group_name, group_atoms in groups.items():
        solute = Solute.from_atoms(group_atoms,
                            {'water' : waters},
                            solute_name=group_name)
        solute.run()
        solutes[group_name] = solute

    return solutes


if __name__ == "__main__":

    analysis_to_run = {'hydrogen_bonds_polymer-polymer': False,
                       'hydrogen_bonds_polymer-water': False,
                       'hydrogen_bonds_water-polymer': False,
                       'hydrogen_bonds_water-water': False,
                       'volume' : False,        
                       'rdfs': False,
                       'diffusion_coefficient': False,
                       'ion_diffusion_coefficient': False,
                       'pore_size_distribution': False,
                       'density_profiles': True, 
                       'coordination' : False,
                       'water_interactions' : False}
    path = './'
    tpr = path + 'prod.tpr'
    xtc = path + 'prod_centered.xtc'

    print(f'Loading {tpr} with trajectory {xtc}')
    u = mda.Universe(tpr, xtc)
    dt = u.trajectory[1].time - u.trajectory[0].time
    time = np.arange(0, u.trajectory.n_frames*dt, dt)  # ps

    # calculate percentage of hydrogen bonds over time
    if analysis_to_run['hydrogen_bonds_polymer-polymer']:
        print('\nCalculating hydrogen bonds: polymer-polymer')
        hbonds_count = hydrogen_bonds(u)

        total_h = u.select_atoms('type hn').n_atoms
        hbonds_percentage = (hbonds_count / total_h) * 100

        # save both count and percentage (two columns) to CSV
        hbonds_csv = path + 'hbonds_polymer-polymer.csv'
        hbonds_out = np.column_stack((time, hbonds_count, hbonds_percentage))
        np.savetxt(hbonds_csv, hbonds_out, delimiter=',', header='time,hbonds_count,hbonds_percentage', comments='', fmt='%.6f')
        print(f'Wrote hbonds count and percentage to {hbonds_csv}')

        print(f'{hbonds_percentage.mean():.2f}% out of {total_h} possible hydrogen bonds')

    if analysis_to_run['hydrogen_bonds_polymer-water']:
        print('\nCalculating hydrogen bonds: polymer-water')
        hbonds_count = hydrogen_bonds(u, acceptors_sel='type OW')

        total_h = u.select_atoms('type hn').n_atoms
        hbonds_percentage = (hbonds_count / total_h) * 100

        # save both count and percentage (two columns) to CSV
        hbonds_csv = path + 'hbonds_polymer-water.csv' # donor-acceptor
        hbonds_out = np.column_stack((time, hbonds_count, hbonds_percentage))
        np.savetxt(hbonds_csv, hbonds_out, delimiter=',', header='time,hbonds_count,hbonds_percentage', comments='', fmt='%.6f')
        print(f'Wrote hbonds count and percentage to {hbonds_csv}')

        print(f'{hbonds_percentage.mean():.2f}% out of {total_h} possible hydrogen bonds')

    if analysis_to_run['hydrogen_bonds_water-polymer']:
        print('\nCalculating hydrogen bonds: water-polymer')
        hbonds_count = hydrogen_bonds(u, donors_sel='type OW', hydrogens_sel='type HW')

        total_h = u.select_atoms('type HW').n_atoms
        hbonds_percentage = (hbonds_count / total_h) * 100

        # save both count and percentage (two columns) to CSV
        hbonds_csv = path + 'hbonds_water-polymer.csv' # donor-acceptor
        hbonds_out = np.column_stack((time, hbonds_count, hbonds_percentage))
        np.savetxt(hbonds_csv, hbonds_out, delimiter=',', header='time,hbonds_count,hbonds_percentage', comments='', fmt='%.6f')
        print(f'Wrote hbonds count and percentage to {hbonds_csv}')

        print(f'{hbonds_percentage.mean():.2f}% out of {total_h} possible hydrogen bonds')

    if analysis_to_run['hydrogen_bonds_water-water']:
        print('\nCalculating hydrogen bonds: water-water')
        hbonds_count = hydrogen_bonds(u, acceptors_sel='type OW', donors_sel='type OW', hydrogens_sel='type HW')

        total_h = u.select_atoms('type HW').n_atoms
        hbonds_percentage = (hbonds_count / total_h) * 100

        # save both count and percentage (two columns) to CSV
        hbonds_csv = path + 'hbonds_water-water.csv' # donor-acceptor
        hbonds_out = np.column_stack((time, hbonds_count, hbonds_percentage))
        np.savetxt(hbonds_csv, hbonds_out, delimiter=',', header='time,hbonds_count,hbonds_percentage', comments='', fmt='%.6f')
        print(f'Wrote hbonds count and percentage to {hbonds_csv}')

        print(f'{hbonds_percentage.mean():.2f}% out of {total_h} possible hydrogen bonds')
    
    if analysis_to_run['volume']:
        # calculate the volume over time
        print('\nCalculating bulk membrane volume')
        volume, xy_area, membrane_bounds = calculate_volume(u)

        # save to CSV
        volume_csv = path + 'membrane_volume.csv'
        vol_out = np.column_stack((time, volume))
        np.savetxt(volume_csv, vol_out, delimiter=',', header='time,membrane_volume_A3', comments='', fmt='%.6f')
        print(f'Wrote membrane volume to {volume_csv}')

        bounds_csv = path + 'membrane_bounds.csv'
        bounds_out = np.column_stack((time, membrane_bounds))
        np.savetxt(bounds_csv, bounds_out, delimiter=',', header='time,membrane_lower_bound_A,membrane_upper_bound_A', comments='', fmt='%.6f')
        print(f'Wrote membrane bounds to {bounds_csv}')

        print(f'Average membrane volume: {volume.mean():.2f} A^3')
        print(f'Average membrane bounds: {membrane_bounds.mean(axis=0)} A')

    if analysis_to_run['rdfs']:

        membrane_bounds = determine_membrane_bounds(u)

        # calculate RDFs
        print('\nCalculating RDFs for water in membrane')
        rdfs = calculate_RDFs(u, membrane_lower_bound=membrane_bounds[1], membrane_upper_bound=membrane_bounds[3], njobs=16)

        # save to CSV
        rdfs_csv = path + 'rdfs.csv'
        rdfs.to_csv(rdfs_csv, index=False)
        print(f'Wrote RDFs to {rdfs_csv}')

    if analysis_to_run['diffusion_coefficient']:
        # calculate MSDs and diffusion coefficients
        print('\nCalculating water MSD and diffusion coefficients in membrane')
        diff = DiffusionCoefficient(tpr, xtc, water='type OW')
        frac, waters = diff.restrict_to_membrane(diff.water, frac=0.8, membrane_bounds=(membrane_bounds[0], membrane_bounds[-1]))
        D, D_ci = diff.run(waters, n_bootstraps=50, confidence=0.5)
        print(f'{len(waters)} waters stay in the membrane')
        save_object(diff, path + 'diffusion_coefficient.pkl')

        # save to MSD to CSV
        msd_csv = path + 'water_msd.csv'
        np.savetxt(msd_csv, np.column_stack((diff.lagtimes, diff.msd_ts, diff.msd_stderr, diff.msd_ci[0], diff.msd_ci[1])), delimiter=',', header='time,msd,stderr,lower_ci,upper_ci', comments='', fmt='%.6f')
        print(f'Wrote water MSD to {msd_csv}')

        print(f'Diffusion coefficient: {D*10**-9:.4e} m^2/s with CI ({D_ci[0]*10**-9:.4e}, {D_ci[1]*10**-9:.4e})')

        # for unrestricted waters
        diff = DiffusionCoefficient(tpr, xtc, water='type OW')
        D, D_ci = diff.run(diff.water, n_bootstraps=50, confidence=0.5)
        save_object(diff, path + 'diffusion_coefficient_unrestricted.pkl')

        # save to MSD to CSV
        msd_csv = path + 'water_msd_unrestricted.csv'
        np.savetxt(msd_csv, np.column_stack((diff.lagtimes, diff.msd_ts, diff.msd_stderr, diff.msd_ci[0], diff.msd_ci[1])), delimiter=',', header='time,msd,stderr,lower_ci,upper_ci', comments='', fmt='%.6f')
        print(f'Wrote water MSD to {msd_csv}')

        print(f'Diffusion coefficient: {D*10**-9:.4e} m^2/s with CI ({D_ci[0]*10**-9:.4e}, {D_ci[1]*10**-9:.4e})')

    if analysis_to_run['ion_diffusion_coefficient']:
        # calculate MSDs and diffusion coefficients
        print('\nCalculating cation MSD and diffusion coefficients')

        # since prod_centered.xtc is already centered, just perform NoJump transformation
        diff = DiffusionCoefficient(None, None)
        diff.universe = mda.Universe(tpr, xtc)
        diff.cation = diff.universe.select_atoms('resname NA')

        workflow = [trans.nojump.NoJump()]
        diff.universe.trajectory.add_transformations(*workflow)

        D, D_ci = diff.run(diff.cation, n_bootstraps=50, confidence=0.5)
        save_object(diff, path + 'ion_diffusion_coefficient.pkl')

        # save to MSD to CSV
        msd_csv = path + 'ion_msd.csv'
        np.savetxt(msd_csv, np.column_stack((diff.lagtimes, diff.msd_ts, diff.msd_stderr, diff.msd_ci[0], diff.msd_ci[1])), delimiter=',', header='time,msd,stderr,lower_ci,upper_ci', comments='', fmt='%.6f')
        print(f'Wrote ion MSD to {msd_csv}')

        print(f'Diffusion coefficient: {D*10**-9:.4e} m^2/s with CI ({D_ci[0]*10**-9:.4e}, {D_ci[1]*10**-9:.4e})')

    if analysis_to_run['pore_size_distribution']:
        # merge pore size distribution data from all frames
        print('\nAveraging pore size distributions')
        frames = natsorted(glob(path + 'poreblazer/frame*/Total_psd.txt'))
        bins, psd, std = average_PSD(frames)

        # save to CSV
        psd_csv = path + 'average_psd.csv'
        np.savetxt(psd_csv, np.column_stack((bins, psd, std)), delimiter=',', header='pore_diameter_A,average_psd,std_dev', comments='', fmt='%.6f')
        print(f'Wrote average PSD to {psd_csv}')

        print(f'Max in the PSD is at {bins[np.argmax(psd)]:.2f} Angstroms')

    if analysis_to_run['density_profiles']:

        # # calculate density profiles and average densities
        # print('\nCalculating density profiles and average densities')
        # bins, water_density_profile, polymer_density_profile, density_df = calculate_density(u, bin_width=1)

        # # save both profiles and time series to CSV
        # density_csv = path + 'density.csv'
        # density_df.to_csv(density_csv, index=False)
        # print(f'Wrote average densities to {density_csv}')

        # density_profile_csv = path + 'density_profiles.csv'
        # density_profile_out = np.column_stack((bins, water_density_profile, polymer_density_profile))
        # np.savetxt(density_profile_csv, density_profile_out, delimiter=',', header='z,water_density_profile,polymer_density_profile', comments='', fmt='%.6f')
        # print(f'Wrote density profiles to {density_profile_csv}')

        membrane_bounds, water_density_profile, polymer_density_profile, bins = determine_membrane_bounds(u, weighted='number', return_hist=True)

        density_profile_csv = path + 'density_profiles_heavy_atoms.csv'
        density_profile_out = np.column_stack((bins, water_density_profile, polymer_density_profile))
        np.savetxt(density_profile_csv, density_profile_out, delimiter=',', header='z,water_density_profile,polymer_density_profile', comments='', fmt='%.6f')
        print(f'Wrote density profiles to {density_profile_csv}')

    if analysis_to_run['coordination']:

        # calculate coordination environment around water molecules
        print('\nCalculating coordination environment around water molecules')
        solute = coordination_analysis(u)

        # save to pickle
        coord_pkl = path + 'water_coordination.pkl'
        save_object(solute, coord_pkl)
        print(f'Wrote water coordination analysis to {coord_pkl}')

    
    if analysis_to_run['water_interactions']:

        # calculate the fraction of water molecules interacting with different functional groups
        print('\nCalculating water interactions with different functional groups')
        solutes = water_interactions(u)

        # save to pickles
        for group_name, solute in solutes.items():
            interaction_pkl = path + f'water_interactions_{group_name}.pkl'
            save_object(solute, interaction_pkl)
            print(f'Wrote water interactions with {group_name} to {interaction_pkl}')

    