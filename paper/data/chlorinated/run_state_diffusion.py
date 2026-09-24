# State-dependent diffusion coefficients to estimate diffusion in membrane only

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
import multiprocessing as mp
from time import perf_counter

import MDAnalysis as mda
import MDAnalysis.transformations as trans


def fit_msd(msds: np.ndarray, lagtimes: np.ndarray):
    '''
    Fit the MSD vs time lags to a line to extract the diffusion coefficient.
    '''
    X = lagtimes
    X = sm.add_constant(X)
    Y = msds

    ols = sm.OLS(Y, X)
    fit_results = ols.fit()
    b, m = fit_results.params
    be, me = fit_results.bse 

    D = m / 6.0  
    stderr = me / 6.0  

    return D, stderr


def fill_gaps(state_array, tol):
    """Fills short gaps of 0s bounded by 1s."""
    if tol == 0:
        return state_array
    filled = state_array.copy()
    zeros_idx = np.where(filled == 0)[0]
    if len(zeros_idx) == 0:
        return filled
    
    gaps = np.split(zeros_idx, np.where(np.diff(zeros_idx) != 1)[0] + 1)
    for gap in gaps:
        if 0 < len(gap) <= tol:
            if gap[0] > 0 and gap[-1] < len(filled) - 1:
                if filled[gap[0]-1] == 1 and filled[gap[-1]+1] == 1:
                    filled[gap] = 1
    return filled


# --- Multiprocessing Worker Setup ---

_shared_states = None
_shared_positions = None

def _init_worker(states, positions):
    """Initialize shared read-only arrays for worker processes to avoid pickling overhead."""
    global _shared_states
    global _shared_positions
    _shared_states = states
    _shared_positions = positions

def _single_bootstrap(b_seed, n_atoms, n_frames, tolerance, dt, start_time, end_time):
    """Performs a single bootstrap iteration."""
    start_perf = perf_counter()

    np.random.seed(b_seed) # Ensure unique RNG state per worker
    
    sum_sq_disp = np.zeros(n_frames, dtype=np.float64)
    count_sq_disp = np.zeros(n_frames, dtype=np.int64)
    
    bootstrapped_indices = np.random.choice(n_atoms, n_atoms, replace=True)

    for atom_idx in bootstrapped_indices:
        atom_states = _shared_states[:, atom_idx]
        if tolerance > 0:
            atom_states = fill_gaps(atom_states, tolerance)
        
        padded = np.pad(atom_states, (1, 1), 'constant', constant_values=(0, 0))
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        
        atom_pos = _shared_positions[:, atom_idx, :] 
        
        for s, e in zip(starts, ends):
            seg_len = e - s
            if seg_len < 2:
                continue 
            
            seg_pos = atom_pos[s:e]
            
            for tau in range(1, seg_len):
                disp = seg_pos[tau:] - seg_pos[:-tau]
                sq_disp = np.sum(disp**2, axis=1)
                
                sum_sq_disp[tau] += np.sum(sq_disp)
                count_sq_disp[tau] += len(sq_disp)

    valid_tau = count_sq_disp > 0
    if not np.any(valid_tau):
        return None, "No valid trajectory segments found in state 1. Check your state definitions."
        
    msd = np.zeros(n_frames)
    msd[valid_tau] = sum_sq_disp[valid_tau] / count_sq_disp[valid_tau]

    times = np.arange(n_frames) * dt
    max_tau_idx = np.where(valid_tau)[0][-1]
    
    if max_tau_idx < 2:
        return None, "Segments are too short to calculate a diffusion coefficient."
        
    start_idx = np.where(times[valid_tau] >= start_time)[0][0]
    end_idx = np.where(times[valid_tau] <= end_time)[0][-1]
    
    tau_fit = times[valid_tau][start_idx:end_idx]
    msd_fit = msd[valid_tau][start_idx:end_idx]
    
    D_val, _ = fit_msd(msd_fit, tau_fit)
    
    end_perf = perf_counter()
    print(f"Bootstrap iteration {b_seed} completed in {end_perf - start_perf:.2f} seconds.")
    return count_sq_disp, msd, D_val, start_idx, end_idx


def compute_state_dependent_diffusion(
    ag: mda.AtomGroup, 
    states: np.ndarray, 
    n_bootstraps: int = 1000,
    dt: float = None, 
    tolerance: int = 2, 
    start_time: float = 150_000, 
    end_time: float = 300_000, 
    dim: int = 3,
    msd_csv: str = "msd_output.csv", 
    D_csv: str = "diffusion_output.csv",
    n_jobs: int = -1 # Added CPU parameter
):
    
    u = ag.universe
    n_frames = u.trajectory.n_frames
    n_atoms = len(ag)

    states = np.asarray(states)
    if states.shape == (n_atoms, n_frames):
        states = states.T
    elif states.shape != (n_frames, n_atoms):
        raise ValueError(f"Expected states array of shape ({n_frames}, {n_atoms}) or ({n_atoms}, {n_frames}). Got {states.shape}.")

    if dt is None:
        dt = u.trajectory.dt
        if dt == 0.0:
            raise ValueError("Trajectory dt is 0. Please provide the 'dt' parameter manually.")

    positions = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    for i, _ in tqdm(enumerate(u.trajectory), desc="Extracting positions", total=n_frames):
        positions[i] = ag.positions

    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    # Generate random seeds for each bootstrap to ensure diverse RNG across processes
    seeds = np.random.randint(0, 10000000, size=n_bootstraps)
    worker_args = [(seed, n_atoms, n_frames, tolerance, dt, start_time, end_time) for seed in seeds]

    count_sq_disp = np.zeros((n_frames, n_bootstraps), dtype=np.int64)
    msd = np.zeros((n_frames, n_bootstraps))
    D = np.zeros(n_bootstraps)

    start_idx, end_idx = 0, 0

    with mp.Pool(processes=n_jobs, initializer=_init_worker, initargs=(states, positions)) as pool:
        results = list(tqdm(
            pool.starmap(_single_bootstrap, worker_args), 
            total=n_bootstraps, 
            desc=f"Bootstrapping (N={n_jobs} cores)"
        ))

    # Reconstruct arrays from pool output
    for b, res in enumerate(results):
        if res[0] is None:
            raise ValueError(res[1]) # Yields error message from worker
            
        counts_b, msd_b, D_val, s_idx, e_idx = res
        count_sq_disp[:, b] = counts_b
        msd[:, b] = msd_b
        D[b] = D_val
        start_idx = s_idx
        end_idx = e_idx

    times = np.arange(n_frames) * dt
    
    # Base validity on the average across all bootstraps (fixes a minor bug in the original logic 
    # where valid_tau was reliant only on the final bootstrap iteration)
    valid_tau = count_sq_disp.sum(axis=1) > 0 

    df_msd = pd.DataFrame({
        'Tau_ps': times[valid_tau],
        'MSD_A2': msd[valid_tau].mean(axis=1),
        'MSD_std_A2': msd[valid_tau].std(axis=1),
        'Samples': count_sq_disp[valid_tau].mean(axis=1),
        'Samples_std': count_sq_disp[valid_tau].std(axis=1)
    })
    
    df_D = pd.DataFrame({
        'Dimensionality': [dim],
        'D_A2_per_ps': [D.mean()],
        'D_std_A2_per_ps': [D.std()],
        'D_cm2_per_s': [D.mean() * 1e-4],
        'Start_Fit_Tau_ps': [times[start_idx]],
        'End_Fit_Tau_ps': [times[end_idx-1]],
        'Fit_Fraction_Used': [(end_idx - start_idx) / len(times)]
    })

    df_msd.to_csv(msd_csv, index=False)
    df_D.to_csv(D_csv, index=False)
    print("Wrote output to:")
    print(f"  MSD CSV: {msd_csv}")
    print(f"  D CSV: {D_csv}")

    return df_msd, df_D


def determine_membrane_bounds(polymer: mda.AtomGroup, water: mda.AtomGroup, weighted: str = 'number', return_hist: bool = False):
    # [Unchanged...]
    u = water.universe
    max_dims = max([ts.dimensions[2] for ts in u.trajectory])

    bin_width = 1
    water_hist = np.histogram([], bins=np.arange(0, max_dims+bin_width, bin_width))[0].astype(np.float64)
    polymer_hist = np.histogram([], bins=np.arange(0, max_dims+bin_width, bin_width))[0].astype(np.float64)

    for ts in tqdm(u.trajectory, desc="Calculating density profiles", total=u.trajectory.n_frames):
        hist, b = np.histogram(water.positions[:,2], bins=np.arange(0, max_dims+bin_width, bin_width), weights=water.masses if weighted == 'mass' else None)
        hist = hist.astype(np.float64) / (bin_width * ts.dimensions[0] * ts.dimensions[1])
        water_hist += hist
        water_bin_centers = 0.5 * (b[:-1] + b[1:])

        hist, b = np.histogram(polymer.positions[:,2], bins=np.arange(0, max_dims+bin_width, bin_width), weights=polymer.masses if weighted == 'mass' else None)
        hist = hist.astype(np.float64) / (bin_width * ts.dimensions[0] * ts.dimensions[1])
        polymer_hist += hist

    water_hist /= u.trajectory.n_frames
    polymer_hist /= u.trajectory.n_frames

    difference = water_hist - polymer_hist
    sign_changes = np.where(np.diff(np.sign(difference)))[0]

    print(f"Found {len(sign_changes)} intersection points:")
    intersections = []
    for idx in sign_changes:
        z1, z2 = water_bin_centers[idx], water_bin_centers[idx + 1]
        d1, d2 = difference[idx], difference[idx + 1]
        
        z_intersect = z1 - d1 * (z2 - z1) / (d2 - d1)
        density_intersect = np.interp(z_intersect, water_bin_centers, water_hist)
        
        intersections.append((z_intersect, density_intersect))
        print(f"  Z = {z_intersect:.2f} Å, Density = {density_intersect:.4f} amu/Å³")

    membrane_bounds = np.linspace(intersections[0][0], intersections[-1][0], 5)
    print(f'Membrane bounds (Z): {membrane_bounds}')

    if return_hist:
        return membrane_bounds, water_hist, polymer_hist
    else:
        return membrane_bounds


def determine_states(water: mda.AtomGroup, membrane_bounds: list):
    # [Unchanged...]
    states = np.zeros((water.universe.trajectory.n_frames, len(water)), dtype=np.int8)
    times = np.zeros(water.universe.trajectory.n_frames, dtype=np.float64)

    for i,ts in tqdm(enumerate(water.universe.trajectory), desc="Determining water states", total=water.universe.trajectory.n_frames):
        z_positions = water.positions[:, 2]
        states[i,:] = (z_positions >= membrane_bounds[0]) & (z_positions <= membrane_bounds[-1])
        times[i] = ts.time

    tmp = {f'water_{j}': states[:, j] for j in range(len(water))}
    df = pd.DataFrame(tmp)
    df['time_ps'] = times
    df.to_csv("water_states.csv", index=False)

    return states


if __name__ == "__main__":

    transformed = 'prod_nojump.xtc'
    output = "middle_membrane.csv"

    u = mda.Universe('prod.tpr', 'prod_centered.xtc')
    PA = u.select_atoms('resname PA*')
    water = u.select_atoms('resname SOL')
    ow = water.select_atoms('type OW')

    membrane_bounds = determine_membrane_bounds(PA, water, weighted='number', return_hist=False)
    states = determine_states(ow, (membrane_bounds[1], membrane_bounds[-2]))

    if not transformed:
        workflow = []
        workflow.append(trans.unwrap(u.atoms))
        workflow.append(trans.center_in_box(PA, center='mass'))
        workflow.append(trans.wrap(u.atoms))
        workflow.append(trans.nojump.NoJump())
        u.trajectory.add_transformations(*workflow)
    else:
        u = mda.Universe('prod.tpr', transformed)
        PA = u.select_atoms('resname PA*')
        water = u.select_atoms('resname SOL')
        ow = water.select_atoms('type OW')

    # Now runs in parallel across CPU threads automatically
    df_msd, df_D = compute_state_dependent_diffusion(
        ag=ow,
        states=states,
        n_bootstraps=50, 
        dt=u.trajectory.dt,
        tolerance=2,
        start_time=150_000,
        end_time=300_000,
        dim=3,
        msd_csv=f"msd_{output}",
        D_csv=f"diffusion_{output}",
        n_jobs=8 # -1 automatically uses all available CPU threads
    )