# track water dwells in the membrane

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import numpy as np
import pandas as pd
import seaborn as sns

import MDAnalysis as mda

membrane_bounds = { # as determined by the intersection of the polymer and water density profiles
    'Pristine': np.array([ 26.50190379,  49.39451567,  72.28712755,  95.17973943,   118.07235131]),
    'NaOCl': np.array([ 24.32690572,  47.28358504,  70.24026436,  93.19694368,   116.153623  ]),
    'NaOCl + NaOH': np.array([ 5.21002256, 28.0388828 , 50.86774304, 73.69660328, 96.52546351])
}

def z_to_region_array(z, lower_bound, upper_bound):
    """
    Convert wrapped z coordinates into region labels for all frames and molecules.

    Parameters
    ----------
    z : np.ndarray, shape (n_frames, n_waters)
        z-coordinate trajectory for all waters.
    lower_bound, upper_bound : float
        Membrane bounds in z.

    Returns
    -------
    np.ndarray, shape (n_frames, n_waters), dtype '<U1'
        Region labels where each element is:
        - 'A' for z > upper_bound
        - 'B' for z < lower_bound
        - 'M' for lower_bound <= z <= upper_bound
    """
    z = np.asarray(z)
    if z.ndim != 2:
        raise ValueError("z must be a 2D array with shape (n_frames, n_waters)")

    regions = np.full(z.shape, "M", dtype="<U1")
    regions[z > upper_bound] = "A"
    regions[z < lower_bound] = "B"
    return regions


def detect_membrane_events(region_array, z=None, Lz=None):
    """
    Detect membrane dwell events from region labels.

    An event is a consecutive run of 'M' for a given molecule.

    Parameters
    ----------
    region_array : np.ndarray, shape (n_frames, n_waters)
        Array of region labels ('A', 'B', 'M').
    z : np.ndarray, optional, shape (n_frames, n_waters)
        Wrapped z coordinates. If provided with Lz, periodic-wrap checks are
        performed and reported per event.
    Lz : float, optional
        z box length used for periodic-wrap detection with threshold
        abs(dz) > 0.5 * Lz.

    Returns
    -------
    list[dict]
        One dictionary per membrane event with keys:
        - mol
        - start_frame
        - end_frame
        - dwell_time
        - entry_region
        - exit_region
        - periodic_jump
    """
    region_array = np.asarray(region_array)
    if region_array.ndim != 2:
        raise ValueError("region_array must be 2D with shape (n_frames, n_waters)")

    valid = np.isin(region_array, ["A", "B", "M"])
    if not np.all(valid):
        raise ValueError("region_array must only contain labels 'A', 'B', and 'M'")

    use_periodic_check = z is not None and Lz is not None
    if use_periodic_check:
        z = np.asarray(z)
        if z.shape != region_array.shape:
            raise ValueError("z must have the same shape as region_array")

    n_frames, n_waters = region_array.shape
    events = []

    for mol in range(n_waters):
        r = region_array[:, mol]
        zm = z[:, mol] if use_periodic_check else None
        t = 0
        while t < n_frames:
            if r[t] != "M":
                t += 1
                continue

            start = t
            while t + 1 < n_frames and r[t + 1] == "M":
                t += 1
            end = t

            entry_region = r[start-1] if start > 0 else None
            exit_region = r[end+1] if end+1 < n_frames else None

            periodic_jump = False
            if use_periodic_check:
                left = max(1, start)
                right = min(n_frames - 1, end+1)
                if right >= left:
                    dz = np.abs(zm[left : right + 1] - zm[left - 1 : right])
                    periodic_jump = bool(np.any(dz > 0.5 * Lz))

            events.append({
                "mol": int(mol),
                "start_frame": int(start),
                "end_frame": int(end),
                "dwell_time": int(end - start + 1),
                "entry_region": None if entry_region is None else str(entry_region),
                "exit_region": None if exit_region is None else str(exit_region),
                "periodic_jump": periodic_jump,
            })
            t += 1

    return events


def label_events(membrane_events_df):
    """
    Add directional and path-level labels to membrane dwell events.

    This function classifies each event using its `(entry_region, exit_region)` pair,
    then maps those directional labels into direction-agnostic `path` categories.
    A post-processing correction is applied so that periodic-wrap artifacts are not
    treated as true crossings.

    Parameters
    ----------
    membrane_events_df : pandas.DataFrame
        Event table produced by `detect_membrane_events` (or equivalent) containing
        at least the following columns:
        - `entry_region` : str or None
            Region immediately before the membrane dwell (`"A"`, `"B"`, or missing).
        - `exit_region` : str or None
            Region immediately after the membrane dwell (`"A"`, `"B"`, or missing).
        - `periodic_jump` : bool
            Whether a periodic boundary jump was detected around the dwell.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame object with two added/updated columns:
        - `direction` : str
            Fine-grained transition label, one of:
            `A_to_B`, `B_to_A`, `A_to_A`, `B_to_B`,
            `inside_to_A`, `inside_to_B`, `A_to_inside`,
            `B_to_inside`, `inside_to_inside`.
        - `path` : str
            Direction-agnostic category, one of:
            `crossing`, `return`, `entry`, `exit`, `trap`, or `unclassified`.

    Notes
    -----
    - Missing values in `entry_region`/`exit_region` are treated as `None`
      before pair mapping.
    - Any event initially labeled as `crossing` with `periodic_jump == True`
      is relabeled to `return` to avoid counting periodic-wrap artifacts as
      physical transmembrane crossings.
    - The function mutates `membrane_events_df` in place and also returns it
      for convenience/chaining.

    Raises
    ------
    KeyError
        If one or more required columns are missing.
    """

    # Create direction labels from entry/exit region pairs.
    pair_to_direction = {
        ("A", "B"): "A_to_B",
        ("B", "A"): "B_to_A",
        ("A", "A"): "A_to_A",
        ("B", "B"): "B_to_B",
        (None, "A"): "inside_to_A",
        (None, "B"): "inside_to_B",
        ("A", None): "A_to_inside",
        ("B", None): "B_to_inside",
        (None, None): "inside_to_inside",
    }

    entry = membrane_events_df["entry_region"].where(pd.notna(membrane_events_df["entry_region"]), None)
    exit_ = membrane_events_df["exit_region"].where(pd.notna(membrane_events_df["exit_region"]), None)

    pairs = pd.Series(list(zip(entry, exit_)), index=membrane_events_df.index)
    membrane_events_df["direction"] = pairs.map(pair_to_direction)

    direction_agnostic_map = {
        "A_to_B": "crossing",
        "B_to_A": "crossing",
        "A_to_A": "return",
        "B_to_B": "return",
        "inside_to_A": "exit",
        "inside_to_B": "exit",
        "A_to_inside": "entry",
        "B_to_inside": "entry",
        "inside_to_inside": "trap",
    }
    membrane_events_df["path"] = membrane_events_df['direction'].map(direction_agnostic_map).fillna('unclassified')
    print(membrane_events_df['path'].unique())

    # if it is labeled as "crossing", but it has a periodic jump, then it is actually a "return"
    mask = (membrane_events_df['periodic_jump']) & (membrane_events_df['path'] == 'crossing')
    membrane_events_df.loc[mask, 'path'] = 'return'

    return membrane_events_df


def save_events(membrane_events_df, path):
    """
    Filter and save a membrane-event table after removing short `return` events.

    This helper applies a percentile-based cleanup to reduce the overwhelming
    number of very short events in the `return` class:
    1) Compute the 99th percentile of `dwell_time` among rows where
       `path == "return"`.
    2) Drop only those `return` rows with `dwell_time` strictly below that
       percentile threshold.
    3) Save the filtered table to CSV at
       `<path>/membrane_events_df_filtered.csv`.

    Parameters
    ----------
    membrane_events_df : pandas.DataFrame
        Input event table that must include at least:
        - `path` : str
            Event category label (e.g., `crossing`, `return`, ...).
        - `dwell_time` : numeric
            Event duration in frames.
    path : str
        Output directory prefix used to build the destination CSV path.
        Expected to point to the treatment folder (for example,
        `./data/pristine/`).

    Returns
    -------
    pandas.DataFrame
        Filtered copy of `membrane_events_df` with short `return` rows removed.

    Side Effects
    ------------
    - Writes a CSV file to `<path>/membrane_events_df_filtered.csv`.
    - Prints the computed percentile threshold, number of dropped rows,
      remaining row count, and output filename.

    Notes
    -----
    - Rows with `path != "return"` are always retained.
    - `return` rows with `dwell_time` equal to the percentile threshold are
      retained (drop condition is strictly `< q99`).
    - The input DataFrame is not modified in place; a filtered copy is created.

    Raises
    ------
    KeyError
        If required columns (`path`, `dwell_time`) are missing.
    OSError
        If the output file cannot be written (e.g., invalid or missing path).
    """
    # Drop short return events: keep only return events at or above the 99th percentile.
    return_mask = membrane_events_df["path"] == "return"
    q99_return = membrane_events_df.loc[return_mask, "dwell_time"].quantile(0.99)

    drop_mask = return_mask & (membrane_events_df["dwell_time"] < q99_return)
    filtered_membrane_events_df = membrane_events_df.loc[~drop_mask].copy()

    out_csv = path + "membrane_events_df_filtered.csv"
    filtered_membrane_events_df.to_csv(out_csv, index=False)

    print(f"Return 99th percentile dwell_time: {q99_return:.3f}")
    print(f"Dropped rows: {int(drop_mask.sum())}")
    print(f"Remaining rows: {filtered_membrane_events_df.shape[0]}")
    print(f"Saved: {out_csv}")

    return filtered_membrane_events_df


def plot_event_trajectory(idx, membrane_events_df, positions, lb=26.5, ub=118, max_x=51, max_z=136):
    """
    Plot the x-z trajectory for one membrane event with start/end highlighting.

    The event is selected by row index from `membrane_events_df`. Coordinates
    are extracted from `positions` using the event molecule index (`mol`) and
    a frame window from `start_frame - 1` through `end_frame + 1` (implemented
    as Python slicing `start : end_frame + 2`, with start clipped at 0).

    Visual encoding:
    - point color: frame order mapped with `matplotlib.cm.coolwarm`
    - first point: `x` marker
    - last point: square marker
    - intermediate points: circle markers
    - membrane bounds (`lb`, `ub`): red dashed vertical lines in z

    Parameters
    ----------
    idx : int
        Row label/index of the event in `membrane_events_df`.
    membrane_events_df : pandas.DataFrame
        Event table containing at least columns:
        `start_frame`, `end_frame`, and `mol`.
    positions : np.ndarray of shape (n_frames, n_waters, 3)
        Wrapped trajectory coordinates for all water molecules.
        Axis meaning: frame, molecule index, Cartesian component
        (`x`, `y`, `z`) respectively.
    lb : float, default=26.5
        Lower membrane boundary in z (A).
    ub : float, default=118
        Upper membrane boundary in z (A).
    max_x : float, default=51
        Upper y-axis limit for plotted x values (A).
    max_z : float, default=136
        Upper x-axis limit for plotted z values (A).

    Returns
    -------
    None
        Displays a Matplotlib figure and prints the selected event row.

    Raises
    ------
    KeyError
        If `idx` is not present or required columns are missing.
    IndexError
        If `mol` is outside the valid index range for `positions`.
    """

    row = membrane_events_df.loc[idx]

    if row['start_frame'] == 0:
        start = 0
    else:
        start = row['start_frame'] - 1

    to_plot = positions[start:row['end_frame']+2, row['mol'], :]
    print(row)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_xlim(0, max_z)
    ax.set_ylim(0, max_x)
    ax.axvline(lb, ls='--', c='r')
    ax.axvline(ub, ls='--', c='r')
    ax.set_aspect('equal')

    n_points = to_plot.shape[0]
    norm_vals = np.linspace(0, 1, n_points)
    colors = plt.cm.coolwarm(norm_vals)

    for i in range(n_points):
        marker = 'o'
        size = 14
        if i == 0:
            marker = 'x'
            size = 40
        elif i == n_points - 1:
            marker = 's'
            size = 28

        ax.scatter(
            to_plot[i, 2],
            to_plot[i, 0],
            s=size,
            marker=marker,
            color=colors[i],
        )

    # Add compact colorbar to show point-order mapping from coolwarm(0) to coolwarm(1).
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([f"Start", f"End"])


def plot_event_z_lines_by_time(event_indices, membrane_events_df, positions, time, ub, lb):
    """
    Plot z-coordinate trajectories for selected membrane events with a shared
    absolute-time colormap.

    Each selected event is drawn as a horizontal line track in z (x-axis), where
    line color encodes simulation time using one global normalization spanning
    ``time[0]`` to ``time[-1]``. This makes colors directly comparable across
    different events: events occurring in similar absolute time windows will have
    similar colors, even if their internal durations differ.

    Parameters
    ----------
    event_indices : array-like of int
        Row positions (``iloc`` indices) in ``membrane_events_df`` identifying
        which events to plot.
    membrane_events_df : pandas.DataFrame
        Event table containing at least:
        - ``start_frame`` : int
        - ``end_frame`` : int
        - ``mol`` : int
    positions : np.ndarray, shape (n_frames, n_molecules, 3)
        Trajectory coordinates. The z-coordinate is read from index 2.
    time : np.ndarray, shape (n_frames,)
        Time values (for example ns) aligned with the first axis of
        ``positions``.
    ub : float
        Upper membrane boundary in z (drawn as a red dashed vertical line).
    lb : float
        Lower membrane boundary in z (drawn as a red dashed vertical line).

    Returns
    -------
    None
        Displays a Matplotlib figure.

    Notes
    -----
    - Frame windows are expanded to ``start_frame - 1`` through
      ``end_frame + 1`` (implemented with Python slice stop ``+2``), then
      clipped to valid frame bounds.
    - Events with fewer than 2 points after clipping are skipped.
    - The y-position of each plotted line corresponds to event order in
      ``event_indices`` (0, 1, 2, ...), not the molecule ID itself.

    Raises
    ------
    ValueError
        If ``event_indices`` is empty, or if no selected events produce
        plottable line segments.
    IndexError
        If an index in ``event_indices`` is out of bounds for
        ``membrane_events_df.iloc``.
    """
    event_indices = np.asarray(event_indices, dtype=int)
    if event_indices.size == 0:
        raise ValueError('event_indices must contain at least one event index.')

    fig, ax = plt.subplots(1, 1, figsize=(10, max(0.1*len(event_indices)+0.5,4)))
    max_z = float(np.nanmax(positions[:, :, 2]))
    ax.set_xlim(0, max_z)
    ax.axvline(lb, ls='--', c='r')
    ax.axvline(ub, ls='--', c='r')
    ax.set_xlabel('z coordinate')
    ax.set_ylabel('molecule index')

    # Use one absolute time scale for all lines and the shared colorbar.
    global_norm = plt.Normalize(vmin=time[0], vmax=time[-1])
    last_line = None
    y_min, y_max = np.inf, -np.inf

    for i,event in enumerate(event_indices):
        start = int(membrane_events_df.loc[event]['start_frame']) - 1
        end = int(membrane_events_df.loc[event]['end_frame']) + 2
        mol = int(membrane_events_df.loc[event]['mol'])

        start = max(0, start)
        end = min(len(time), end)
        if end - start < 2:
            continue

        to_plot = positions[start:end, mol, :]
        event_time = time[start:end]

        x = to_plot[:, 2]
        y = np.full_like(x, i, dtype=float)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=plt.cm.viridis, norm=global_norm, linewidth=2.0, alpha=0.9)
        lc.set_array(event_time[:-1])
        last_line = ax.add_collection(lc)

        # Add a directional arrow at the end of each line.
        end_color = plt.cm.viridis(global_norm(event_time[-1]))
        ax.annotate(
            '',
            xy=(x[-1], y[-1]),
            xytext=(x[-2], y[-2]),
            arrowprops=dict(arrowstyle='->', color=end_color, lw=2.0, shrinkA=0, shrinkB=0),
        )

        y_min = min(y_min, np.min(y))
        y_max = max(y_max, np.max(y))

    if last_line is None:
        raise ValueError('No plottable segments were generated from event_indices.')

    pad = 1.0
    ax.set_ylim(y_min - pad, y_max + pad)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=global_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Time (ns)')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    label = 'NaOCl + NaOH'
    path = "./"

    u = mda.Universe(path + 'prod.tpr', path + 'prod_centered.xtc')
    selection = 'type OW'
    water = u.select_atoms(selection)
    n_frames = len(u.trajectory)

    # Get positions from trajectory starting at frame 0
    positions = []
    max_x, max_y, max_z = 0, 0, 0
    for ts in u.trajectory:
        positions.append(water.positions)

        if ts.dimensions[0] > max_x:
            max_x = ts.dimensions[0]
        if ts.dimensions[1] > max_y:
            max_y = ts.dimensions[1]
        if ts.dimensions[2] > max_z:
            max_z = ts.dimensions[2]

    positions = np.asarray(positions)

    lb, ub = membrane_bounds[label][0], membrane_bounds[label][-1]
    Lz = u.dimensions[2]

    print(f"Max box dimensions: x={max_x:.2f} Å, z={max_z:.2f} Å")

    region_array = z_to_region_array(positions[:, :, 2], lb, ub)
    membrane_events = detect_membrane_events(region_array, z=positions[:, :, 2], Lz=Lz)

    membrane_events_df = pd.DataFrame(membrane_events)
    membrane_events_df = label_events(membrane_events_df)
    print(membrane_events_df.shape)

    save_events(membrane_events_df, path)
