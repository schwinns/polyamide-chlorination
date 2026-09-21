# State-dependent diffusion coefficients to estimate diffusion in membrane only

import numpy as np
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm

import MDAnalysis as mda
import MDAnalysis.transformations as trans

def compute_state_dependent_diffusion(
    ag: mda.AtomGroup, 
    states: np.ndarray, 
    dt: float = None, 
    tolerance: int = 2, 
    fit_fraction: float = 0.5, 
    dim: int = 3,
    msd_csv: str = "msd_output.csv", 
    D_csv: str = "diffusion_output.csv"
):
    """
    Computes state-dependent diffusion coefficients from MDAnalysis trajectories.

    Parameters
    ----------
    ag : MDAnalysis.core.groups.AtomGroup
        The AtomGroup containing the particles of interest (e.g., water molecules).
    states : array_like
        A 2D array of states (1 for in-state/membrane, 0 for out-of-state/bulk).
        Expected shape is (n_frames, n_atoms) or (n_atoms, n_frames).
    dt : float, optional
        Time step between frames in picoseconds. If None, it is extracted from the trajectory.
    tolerance : int, default=2
        Maximum number of consecutive frames a particle can be in state 0 
        (due to boundary fluctuation) and still be considered part of the continuous state 1 segment.
    fit_fraction : float, default=0.5
        Fraction of the maximum lag time to use for linear fitting of the diffusion coefficient. 
        Discards the long lag times where statistics are poor.
    dim : int, default=3
        Dimensionality of the diffusion (usually 3 for bulk/isotropic, 2 for lateral).
    msd_csv : str, default="msd_output.csv"
        File path to save the Mean Squared Displacement (MSD) data.
    D_csv : str, default="diffusion_output.csv"
        File path to save the computed diffusion coefficient data.

    Returns
    -------
    df_msd : pandas.DataFrame
        DataFrame containing lag times (Tau_ps), MSD values (MSD_A2), and sample counts.
    df_D : pandas.DataFrame
        DataFrame containing the calculated Diffusion Coefficient and fit statistics.
    """
    
    u = ag.universe
    n_frames = u.trajectory.n_frames
    n_atoms = len(ag)

    # Ensure states matrix is shape (n_frames, n_atoms)
    states = np.asarray(states)
    if states.shape == (n_atoms, n_frames):
        states = states.T
    elif states.shape != (n_frames, n_atoms):
        raise ValueError(f"Expected states array of shape ({n_frames}, {n_atoms}) or ({n_atoms}, {n_frames}). Got {states.shape}.")

    # Get time step
    if dt is None:
        dt = u.trajectory.dt
        if dt == 0.0:
            raise ValueError("Trajectory dt is 0. Please provide the 'dt' parameter manually.")

    # Extract all positions into memory (Assumes trajectory is already unwrapped to avoid PBC artifacts)
    positions = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    for i, _ in tqdm(enumerate(u.trajectory), desc="Extracting positions"):
        positions[i] = ag.positions

    sum_sq_disp = np.zeros(n_frames, dtype=np.float64)
    count_sq_disp = np.zeros(n_frames, dtype=np.int64)

    def fill_gaps(state_array, tol):
        """Fills short gaps of 0s bounded by 1s."""
        if tol == 0:
            return state_array
        filled = state_array.copy()
        zeros_idx = np.where(filled == 0)[0]
        if len(zeros_idx) == 0:
            return filled
        
        # Split into consecutive blocks of 0s
        gaps = np.split(zeros_idx, np.where(np.diff(zeros_idx) != 1)[0] + 1)
        for gap in gaps:
            if 0 < len(gap) <= tol:
                # Ensure the gap is strictly bounded by 1s on both ends
                if gap[0] > 0 and gap[-1] < len(filled) - 1:
                    if filled[gap[0]-1] == 1 and filled[gap[-1]+1] == 1:
                        filled[gap] = 1
        return filled

    # Iterate over each atom to extract valid trajectory segments
    for atom_idx in tqdm(range(n_atoms), desc="Computing MSDs"):
        atom_states = states[:, atom_idx]
        if tolerance > 0:
            atom_states = fill_gaps(atom_states, tolerance)
            
        # Find contiguous segments where state == 1
        padded = np.pad(atom_states, (1, 1), 'constant', constant_values=(0, 0))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        
        atom_pos = positions[:, atom_idx, :] 
        
        for s, e in zip(starts, ends):
            seg_len = e - s
            if seg_len < 2:
                continue 
            
            seg_pos = atom_pos[s:e]
            
            # Compute squared displacements for all possible lag times within this segment
            for tau in range(1, seg_len):
                disp = seg_pos[tau:] - seg_pos[:-tau]
                sq_disp = np.sum(disp**2, axis=1)
                
                sum_sq_disp[tau] += np.sum(sq_disp)
                count_sq_disp[tau] += len(sq_disp)

    # Average MSD computation
    valid_tau = count_sq_disp > 0
    if not np.any(valid_tau):
        raise ValueError("No valid trajectory segments found in state 1. Check your state definitions.")
        
    msd = np.zeros(n_frames)
    msd[valid_tau] = sum_sq_disp[valid_tau] / count_sq_disp[valid_tau]
    times = np.arange(n_frames) * dt

    # Fit Diffusion Coefficient (D)
    max_tau_idx = np.where(valid_tau)[0][-1]
    if max_tau_idx < 2:
        raise ValueError("Segments are too short to calculate a diffusion coefficient.")
        
    fit_end = max(2, int(max_tau_idx * fit_fraction))
    
    tau_fit = times[1:fit_end]
    msd_fit = msd[1:fit_end]
    
    slope, intercept, r_value, p_value, std_err = linregress(tau_fit, msd_fit)
    
    # Calculate D: MSD = 2 * dim * D * tau
    # Default units: Angstroms^2 / picosecond (multiply by 1e-4 to get cm^2/s)
    D = slope / (2 * dim)

    # Prepare output DataFrames
    df_msd = pd.DataFrame({
        'Tau_ps': times[valid_tau],
        'MSD_A2': msd[valid_tau],
        'Samples': count_sq_disp[valid_tau]
    })
    
    df_D = pd.DataFrame({
        'Dimensionality': [dim],
        'D_A2_per_ps': [D],
        'D_cm2_per_s': [D * 1e-4],
        'R_squared': [r_value**2],
        'Max_Fit_Tau_ps': [times[fit_end-1]],
        'Fit_Fraction_Used': [fit_fraction]
    })

    # Save to CSV
    df_msd.to_csv(msd_csv, index=False)
    df_D.to_csv(D_csv, index=False)

    return df_msd, df_D


def determine_membrane_bounds(polymer: mda.AtomGroup, water: mda.AtomGroup, weighted: str = 'number', return_hist: bool = False):
    '''
    Determine membrane bounds based on intersection of the polymer desnity and water density profiles
    
    Parameters
    ----------
    polymer : MDAnalysis AtomGroup
        AtomGroup containing the polymer atoms (e.g., selection string = 'resname PA*').
    water : MDAnalysis AtomGroup
        AtomGroup containing the water atoms (e.g., selection string = 'resname SOL').
    weighted : str, optional
        Weighting scheme for the density calculation ('number' or 'mass').
    return_hist : bool, optional
        If True, return the density histograms along with the membrane bounds.

    Returns
    -------
    membrane_bounds : np.ndarray
        Array containing the Z-coordinates of the membrane bounds, bounds are split
        into 4 regions 0-25%, 25-50%, 50-75%, and 75-100% of membrane thickness.
    water_hist : np.ndarray, optional
        Density histogram of water along the Z-axis (only returned if return_hist is True).
    polymer_hist : np.ndarray, optional
        Density histogram of polymer along the Z-axis (only returned if return_hist is True).
    '''

    u = water.universe

    max_dims = max([ts.dimensions[2] for ts in u.trajectory])

    bin_width = 1
    water_hist = np.histogram([], bins=np.arange(0, max_dims+bin_width, bin_width))[0].astype(np.float64)
    polymer_hist = np.histogram([], bins=np.arange(0, max_dims+bin_width, bin_width))[0].astype(np.float64)

    for ts in tqdm(u.trajectory, desc="Calculating density profiles"):
        # Recalculate histograms
        hist, b = np.histogram(water.positions[:,2], bins=np.arange(0, max_dims+bin_width, bin_width), weights=water.masses if weighted == 'mass' else None)
        hist = hist.astype(np.float64) / (bin_width * ts.dimensions[0] * ts.dimensions[1])
        water_hist += hist
        water_bin_centers = 0.5 * (b[:-1] + b[1:])

        hist, b = np.histogram(polymer.positions[:,2], bins=np.arange(0, max_dims+bin_width, bin_width), weights=polymer.masses if weighted == 'mass' else None)
        hist = hist.astype(np.float64) / (bin_width * ts.dimensions[0] * ts.dimensions[1])
        polymer_hist += hist

    # Average over frames
    water_hist /= u.trajectory.n_frames
    polymer_hist /= u.trajectory.n_frames

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

    membrane_bounds = np.linspace(intersections[0][0], intersections[-1][0], 5) # split membrane into quarters
    print(f'Membrane bounds (Z): {membrane_bounds}')

    if return_hist:
        return membrane_bounds, water_hist, polymer_hist
    else:
        return membrane_bounds


def determine_states(water: mda.AtomGroup, membrane_bounds: list):
    '''
    Determine the state of each water molecule based on its position relative to the membrane bounds.

    Parameters
    ----------
    water : MDAnalysis AtomGroup
        AtomGroup containing the water molecules (e.g., selection string = 'resname SOL').
    membrane_bounds : array-like
        Array containing the Z-coordinates of the membrane bounds, should be length 2 [lower_bound, upper_bound].
    '''

    states = np.zeros((water.universe.trajectory.n_frames, len(water)), dtype=np.int8)
    times = np.zeros(water.universe.trajectory.n_frames, dtype=np.float64)

    for i,ts in tqdm(enumerate(water.universe.trajectory), desc="Determining water states"):
        z_positions = water.positions[:, 2]
        states[i,:] = (z_positions >= membrane_bounds[0]) & (z_positions <= membrane_bounds[-1])
        times[i] = ts.time

    # Prepare output DataFrames
    tmp = {f'water_{j}': states[:, j] for j in range(len(water))}
    df = pd.DataFrame(tmp)
    df['time_ps'] = times
    df.to_csv("water_states.csv", index=False)

    return states


if __name__ == "__main__":

    transformed = 'prod_nojump.xtc' # either None or the name of the nojump trajectory
    output = "middle_membrane.csv"

    u = mda.Universe('prod.tpr', 'prod_centered.xtc')
    PA = u.select_atoms('resname PA*')
    water = u.select_atoms('resname SOL')
    ow = water.select_atoms('type OW')

    ###### BEFORE UNWRAPPING TRAJECTORY ######
    # get geometric membrane boundaries
    membrane_bounds = determine_membrane_bounds(PA, water, weighted='number', return_hist=False)

    # get the time series of states for each water molecule
    states = determine_states(ow, (membrane_bounds[1], membrane_bounds[-2]))

    ###### UNWRAPPING ######
    if not transformed:
        # create a workflow for on-the-fly transformations
        workflow = []

        workflow.append(trans.unwrap(u.atoms))
        workflow.append(trans.center_in_box(PA, center='mass'))
        workflow.append(trans.wrap(u.atoms))

        # no jump trajectory unwrapping, necessary for diffusion coefficient calculations
        workflow.append(trans.nojump.NoJump())

        # add transformations
        u.trajectory.add_transformations(*workflow)
    else:
        u = mda.Universe('prod.tpr', transformed)
        PA = u.select_atoms('resname PA*')
        water = u.select_atoms('resname SOL')
        ow = water.select_atoms('type OW')

    ###### MSD AND DIFFUSION COEFFICIENT CALCULATION ######
    df_msd, df_D = compute_state_dependent_diffusion(
        ag=ow,
        states=states,
        dt=u.trajectory.dt,
        tolerance=2,
        fit_fraction=0.5,
        dim=3,
        msd_csv=f"msd_{output}",
        D_csv=f"diffusion_{output}"
    )


    
