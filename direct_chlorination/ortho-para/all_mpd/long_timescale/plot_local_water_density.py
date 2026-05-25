# Plot the 2D water density projections around a give atom

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.colors as mcolors
from matplotlib.collections import EllipseCollection
import matplotlib.gridspec as gridspec

import MDAnalysis as mda
from MDAnalysis.analysis.distances import capped_distance
import MDAnalysis.transformations as trans


# Canonical element colors (common in chemistry visualizations)
element_colors = {
    'C': '#909090',   # Gray
    'N': '#3050F8',   # Blue
    'O': '#FF0D0D',   # Red
    'H': '#FFFFFF',   # White
    'Cl': '#1FF01F',  # Green
    'Na': '#AB5CF2',  # Purple
}

# van der Waals radii in Å (Bondi/Jmol style)
vdw_radii = {
    'H': 1.20,
    'C': 1.70,
    'N': 1.55,
    'O': 1.52,
    'F': 1.47,
    'Na': 2.27,
    'Cl': 1.75,
    'S': 1.80,
    'P': 1.80,
}

def transform_to_plane(positions, point1, point2, point3):
    """
    Transform coordinates so that 3 points define the x-y plane.
    
    Parameters:
    -----------
    positions : array-like, shape (N, 3)
        The coordinates to transform
    point1, point2, point3 : array-like, shape (3,)
        Three points that define the desired x-y plane
    
    Returns:
    --------
    transformed : array, shape (N, 3)
        Coordinates in the new reference frame
    """
    # Convert to numpy arrays
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    
    # Define new basis vectors
    # x-axis: from p1 to p2
    x_axis = p2 - p1
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # z-axis: perpendicular to plane (cross product)
    v2 = p3 - p1
    z_axis = np.cross(x_axis, v2)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # y-axis: perpendicular to both x and z
    y_axis = np.cross(z_axis, x_axis)
    
    # Create rotation matrix (basis vectors as rows)
    rotation_matrix = np.array([x_axis, y_axis, z_axis])
    
    # Center coordinates on p1 (origin of new frame)
    centered = positions - p1
    
    # Apply rotation
    transformed = centered @ rotation_matrix.T
    
    return transformed


def add_plane_aligned_coords(df):
    """For each frame, find 3 closest atoms to Cl and rotate so they define the xy-plane."""
    frames = []
    for frame, g in df.groupby('frame'):
        if len(g) < 3:
            continue  # not enough atoms to define a plane
        anchors = g.nsmallest(3, 'distance_to_atom')[['x', 'y', 'z']]
        p1, p2, p3 = anchors.values  # ordered closest → farther
        coords = g[['x', 'y', 'z']].values
        rotated = transform_to_plane(coords, p1, p2, p3)

        g_out = g.copy()
        g_out[['x_plane', 'y_plane', 'z_plane']] = rotated
        # optional: mark which atoms defined the plane
        g_out['plane_anchor'] = False
        g_out.loc[anchors.index, 'plane_anchor'] = True
        frames.append(g_out)
    return pd.concat(frames, ignore_index=True)


def add_alpha(hex_color, alpha):
    r, g, b = mcolors.to_rgb(hex_color)
    return (r, g, b, alpha)


def plot_atoms(x, y, alpha, elements, radii, ax=None, xlabel='x', ylabel='y', visualization_cutoff=5, atom_indices=None, bonded_indices=None):
    '''
    Plot a 2D projection of atoms as circles with optional bonds
    
    Parameters:
    -----------
    x, y : array-like
        2D coordinates of atoms
    alpha : array-like
        Values that map to transparency
    elements : array-like
        Element types for coloring
    radii : array-like
        Radii for circle sizes
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    xlabel, ylabel : str
        Axis labels
    visualization_cutoff : float
        Axis limits
    atom_indices : array-like, optional
        Indices of atoms corresponding to the coordinates
    bonded_indices : list of lists, optional
        Lists of bonded atom indices for each atom, used for bond visualization
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
    '''

    vmin, vmax = alpha.min(), alpha.max()
    alphas = 0.2 + 0.8 * (1 - (alpha - vmin) / (vmax - vmin))  # closer → more opaque

    # build per-point RGBA using your element palette
    colors = [add_alpha(element_colors[e], a) for e, a in zip(elements, alphas)]

    if ax is None:
        fig, ax = plt.subplots()

    # Draw bonds first (so they appear behind atoms)
    if bonded_indices is not None and atom_indices is not None:
        # Create a mapping from atom indices to position indices
        idx_to_pos = {idx: i for i, idx in enumerate(atom_indices)}

        # Draw lines between bonded atoms
        for atom_idx, bonded_idx_list in zip(atom_indices,  bonded_indices):
            for bonded_idx in bonded_idx_list:
                if bonded_idx in idx_to_pos:
                    pos_i = idx_to_pos[atom_idx]
                    pos_j = idx_to_pos[bonded_idx]
                    ax.plot([x[pos_i], x[pos_j]], [y[pos_i], y[pos_j]], 'k-', linewidth=1.0, alpha=0.6, zorder=1)

    ec = EllipseCollection(
        widths=2*radii, heights=2*radii, angles=0,
        units='xy',                # key: sizes in data coords
        offsets=np.column_stack((x, y)),
        transOffset=ax.transData,
        facecolors=colors,
        edgecolors='k', linewidths=0.8,
        zorder=2  # atoms on top of bonds
    )
    ax.add_collection(ec)
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(-visualization_cutoff, visualization_cutoff+1, 1))
    ax.set_yticks(np.arange(-visualization_cutoff, visualization_cutoff+1, 1))
    ax.set_xlim(-visualization_cutoff, visualization_cutoff)
    ax.set_ylim(-visualization_cutoff, visualization_cutoff)

    return ax


def get_atom_group(selection, universe):
    '''Get the atom group based on selection string, one of 'cl', 'ortho-ortho ha', 'ortho-para ha', 'coo', 'cooh', 'amide hn', 'amide o', 'amine hn' '''

    if selection == 'cl':
        ag = universe.select_atoms('type cl')

    elif selection == 'ortho-ortho ha':
        # look at the ha in ortho-ortho for comparison in pristine
        ca_bonded_n = universe.select_atoms('(type ca) and (bonded type n)') # force it to only be linear MPD fragments
        tmp = universe.select_atoms('(type ca) and (bonded group tmp)', tmp=ca_bonded_n)
        ca_ortho_ortho = []
        for atom in tmp:
            n_bonded_n = len([a for a in atom.bonded_atoms if a in ca_bonded_n])
            if n_bonded_n == 2:
                ca_ortho_ortho.append(atom)

        ca_ortho_ortho = mda.AtomGroup(ca_ortho_ortho)
        ag = universe.select_atoms('type ha and bonded group ca_ortho_ortho', ca_ortho_ortho=ca_ortho_ortho)

    elif selection == 'ortho-para ha':
        # look at the ha in ortho-para for comparison in pristine
        ca_bonded_n = universe.select_atoms('(type ca) and (bonded type n)') 
        tmp = universe.select_atoms('(type ca) and (bonded group tmp)', tmp=ca_bonded_n)
        ca_ortho_para = []
        for atom in tmp:
            n_bonded_n = len([a for a in atom.bonded_atoms if a in ca_bonded_n])
            if n_bonded_n == 1:
                ca_ortho_para.append(atom)

        ca_ortho_para = mda.AtomGroup(ca_ortho_para)
        ag = universe.select_atoms('type ha and bonded group ca_ortho_para', ca_ortho_para=ca_ortho_para)

    elif selection == 'coo':
        coo_c = universe.select_atoms(f'(type c) and (not bonded type oh n)')
        ag = universe.select_atoms(f'(type o) and (bonded group coo_c)', coo_c=coo_c)

    elif selection == 'cooh':
        ag = universe.select_atoms(f'type oh')

    elif selection == 'amide hn':
        xlink_n = universe.select_atoms(f'(type n) and (bonded type c)')
        ag = universe.select_atoms(f'(type hn) and (bonded group xlink_n)', xlink_n=xlink_n)

    elif selection == 'amide o':
        xlink_c = universe.select_atoms(f'(type c) and (bonded type n)')
        ag = universe.select_atoms(f'(type o) and (bonded group xlink_c)', xlink_c=xlink_c)

    elif selection == 'amine hn':
        nh2 = universe.select_atoms(f'type nh')
        ag = universe.select_atoms(f'(type hn) and (bonded group nh2)', nh2=nh2)

    else:
        raise ValueError(f'Unknown selection: {selection}')

    return ag


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot local water density around a selected atom')
    parser.add_argument('--path', type=str, default='./', help='Path to the directory containing the trajectory files')
    parser.add_argument('--tpr', type=str, default='prod.tpr', help='Name of the .tpr file')
    parser.add_argument('--xtc', type=str, default='prod_centered.xtc', help='Name of the .xtc file')
    parser.add_argument('--selection', type=str, default='cl', help='Selection around which to compute density, one of "cl", "ortho-ortho ha", "ortho-para ha", "coo", "cooh", "amide hn", "amide o", "amine hn"')
    parser.add_argument('--idx', type=int, default=0, help='Index of the selected atom to analyze')
    parser.add_argument('--distance_cutoff', type=float, default=8.0, help='Distance cutoff in angstroms for capped_distance calculations')
    parser.add_argument('--z_threshold', type=float, default=2.25, help='Z-plane threshold in angstroms for filtering water density')
    parser.add_argument('--visualization_cutoff', type=float, default=5.0, help='Cutoff in angstroms for plotting nearby atoms')
    parser.add_argument('--density', choices=['kde', 'hist'], default='kde', help='Use KDE or 2D histogram for density plots')
    parser.add_argument('--panels', choices=['all', 'left'], default='all', help='Show all three panels or just the large left panel (for histogram mode)')
    parser.add_argument('--vmin', type=float, default=None, help='Minimum value for colorbar (auto-calculated if not specified)')
    parser.add_argument('--vmax', type=float, default=None, help='Maximum value for colorbar (auto-calculated if not specified)')
    args = parser.parse_args()

    # inputs
    path = args.path
    tpr = path + args.tpr
    xtc = path + args.xtc
    selection = args.selection
    idx = args.idx
    distance_cutoff = args.distance_cutoff
    z_threshold = args.z_threshold
    visualization_cutoff = args.visualization_cutoff
    density_mode = args.density

    print(f'Loading {tpr} with trajectory {xtc}')
    u = mda.Universe(tpr, xtc)
    ag = get_atom_group(selection, u)
    my_atom = ag[idx]

    # center on my_atom
    workflow = [
        trans.center_in_box(mda.AtomGroup([my_atom])),
        trans.wrap(u.atoms)
    ]
    u.trajectory.add_transformations(*workflow)

    # get the coordinates and distances for all frames
    df = pd.DataFrame(columns=['x', 'y', 'z', 'distance_to_atom', 'frame'])
    df_cl = pd.DataFrame(columns=['x', 'y', 'z', 'frame'])
    for ts in u.trajectory:

        pairs, distances = capped_distance(my_atom.position, u.atoms, max_cutoff=distance_cutoff, box=u.dimensions)
        my_neighbors = u.atoms[pairs[:,1]]
        tmp = pd.DataFrame(my_neighbors.positions, columns=['x', 'y', 'z'], index=my_neighbors.indices)
        tmp['element'] = my_neighbors.elements
        tmp['type'] = my_neighbors.types
        tmp['distance_to_atom'] = distances[(pairs[:,0] == 0)]
        tmp['frame'] = u.trajectory.frame
        tmp['my_atom_index'] = my_neighbors.indices
        tmp['bonded_indices'] = [atom.bonded_atoms.indices for atom in my_neighbors]
        df = pd.concat([df, tmp])

    # transform coordinates to plane defined by 3 closest atoms to my_atom in each frame
    df = add_plane_aligned_coords(df)
    df['vdw_radius'] = df['element'].map(vdw_radii)

    # first, compare the density plots with and without z filtering
    frame0 = df.query(f'frame == 0 and type not in ["HW", "OW"] and distance_to_atom <= {visualization_cutoff}').copy()
    u.trajectory[0]  # set to first frame for distance calculations
    pairs,_ = capped_distance(my_atom.position, u.atoms, max_cutoff=distance_cutoff, box=u.dimensions)
    my_neighbors = u.atoms[pairs[:,1]]

    # pick a value to map to transparency (here z_plane) and normalize to [0.2, 1.0]
    alpha = frame0['z_plane']
    x = frame0['x_plane'].to_numpy()
    y = frame0['y_plane'].to_numpy()
    r = frame0['vdw_radius'].to_numpy() * 0.5  # radius in data units, scaled to 50%

    # plot water density and atoms
    df_ow = df.query("type == 'OW'").copy()
    df_ow_filtered = df_ow[np.abs(df_ow['z_plane']) <= z_threshold]

    print(f"Original OW atoms: {df_ow.shape[0]}")
    print(f"Filtered OW atoms (|z| <= {z_threshold}): {df_ow_filtered.shape[0]}")

    # Original
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(data=df_ow, x='x_plane', y='y_plane', bins=50, stat='count', cmap='Blues', ax=ax1, cbar=True)
    ax1.set_title('All OW atoms')
    ax1.set_xlim(-visualization_cutoff, visualization_cutoff); ax1.set_ylim(-visualization_cutoff, visualization_cutoff)
    ax1.set_xticks(np.arange(-visualization_cutoff, visualization_cutoff+1, 1)); ax1.set_yticks(np.arange(-visualization_cutoff, visualization_cutoff+1, 1))
    ax1.set_aspect('equal')

    # Filtered by z_plane
    sns.histplot(data=df_ow_filtered, x='x_plane', y='y_plane', bins=50, stat='count', cmap='Blues', ax=ax2, cbar=True)
    ax2.set_title(f'OW atoms with |z| ≤ {z_threshold} Å')
    ax2.set_xlim(-visualization_cutoff, visualization_cutoff); ax2.set_ylim(-visualization_cutoff, visualization_cutoff)
    ax2.set_xticks(np.arange(-visualization_cutoff, visualization_cutoff+1, 1)); ax2.set_yticks(np.arange(-visualization_cutoff, visualization_cutoff+1, 1))
    ax2.set_aspect('equal')

    sns.scatterplot(data=frame0, x='x_plane', y='y_plane', hue='element', palette=element_colors, s=100, edgecolor='k', ax=ax1)
    sns.scatterplot(data=frame0, x='x_plane', y='y_plane', hue='element', palette=element_colors, s=100, edgecolor='k', ax=ax2)

    plt.tight_layout()
    plt.show()

    # Now make the 3-panel figure with z-filtered water density
    if args.panels == 'all':
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 1])

        # Left panel (spans both rows)
        ax_left = fig.add_subplot(gs[:, 0])

        # Right panels (stacked)
        ax_right_top = fig.add_subplot(gs[0, 1])
        ax_right_bottom = fig.add_subplot(gs[1, 1])
    else:  # args.panels == 'left'
        fig, ax_left = plt.subplots(figsize=(6, 6))
        ax_right_top = None
        ax_right_bottom = None

    # Data setup
    df_ow = df.query("type == 'OW'").copy()
    frame0 = df.query(f'frame == 0 and type not in ["HW", "OW"] and distance_to_atom <= {visualization_cutoff}').copy()
    atom_indices = frame0['my_atom_index'].to_numpy()
    bonded_indices = frame0['bonded_indices'].to_numpy()
    x = frame0['x_plane'].to_numpy()
    y = frame0['y_plane'].to_numpy()
    z = frame0['z_plane'].to_numpy()
    r = frame0['vdw_radius'].to_numpy() * 0.5
    df_ow_filtered = df_ow[np.abs(df_ow['z_plane']) <= z_threshold]

    if density_mode == 'kde':
        vmin = args.vmin if args.vmin is not None else 0
        vmax = args.vmax if args.vmax is not None else 1
        levels = np.linspace(vmin, vmax, 50)

        alpha_xy = frame0['z_plane']
        sns.kdeplot(data=df_ow_filtered, x='x_plane', y='y_plane', fill=True, cmap='Blues', ax=ax_left, levels=levels, thresh=0)
        plot_atoms(x, y, alpha_xy, frame0['element'], r, ax=ax_left, xlabel='x', ylabel='y', visualization_cutoff=visualization_cutoff, atom_indices=atom_indices, bonded_indices=bonded_indices)
        ax_left.text(0.98, 0.98, ha='right', va='top', transform=ax_left.transAxes,
                    s=f'Water |z| ≤ {z_threshold} Å of Cl', fontsize=12)

        if args.panels == 'all':
            # Right top: x-z projection
            alpha_xz = frame0['y_plane']
            sns.kdeplot(data=df_ow, x='x_plane', y='z_plane', fill=True, cmap='Blues', ax=ax_right_top, levels=levels, thresh=0)
            plot_atoms(x, z, alpha_xz, frame0['element'], r, ax=ax_right_top, xlabel='x', ylabel='z', visualization_cutoff=visualization_cutoff, atom_indices=atom_indices, bonded_indices=bonded_indices)
            ax_right_top.tick_params(axis='both', labelsize=8)

            # Right bottom: y-z projection
            alpha_yz = frame0['x_plane']
            sns.kdeplot(data=df_ow, x='y_plane', y='z_plane', fill=True, cmap='Blues', ax=ax_right_bottom, levels=levels, thresh=0)
            plot_atoms(y, z, alpha_yz, frame0['element'], r, ax=ax_right_bottom, xlabel='y', ylabel='z', visualization_cutoff=visualization_cutoff, atom_indices=atom_indices, bonded_indices=bonded_indices)
            ax_right_bottom.tick_params(axis='both', labelsize=8)

        sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        if args.panels == 'all':
            cbar = plt.colorbar(sm, ax=[ax_left, ax_right_top, ax_right_bottom], location='top', shrink=0.8, pad=0.02)
        else:
            cbar = plt.colorbar(sm, ax=ax_left, location='right', shrink=0.8, pad=0.02)
        cbar.set_label('KDE density')

    else:  # density_mode == 'hist'
        hist_bins = 75
        hist_range = [[-visualization_cutoff, visualization_cutoff], [-visualization_cutoff, visualization_cutoff]]

        h_xy, xe_xy, ye_xy = np.histogram2d(df_ow_filtered['x_plane'], df_ow_filtered['y_plane'], bins=hist_bins, range=hist_range)
        h_xz, xe_xz, ze_xz = np.histogram2d(df_ow['x_plane'], df_ow['z_plane'], bins=hist_bins, range=hist_range)
        h_yz, ye_yz, ze_yz = np.histogram2d(df_ow['y_plane'], df_ow['z_plane'], bins=hist_bins, range=hist_range)

        if args.vmax is None:
            vmax = max(h_xy.max(), h_xz.max(), h_yz.max()) if max(h_xy.max(), h_xz.max(), h_yz.max()) > 0 else 1
        else:
            vmax = args.vmax
        vmin = args.vmin if args.vmin is not None else 0

        pcm_xy = ax_left.pcolormesh(xe_xy, ye_xy, h_xy.T, cmap='Blues', shading='auto', vmin=vmin, vmax=vmax)
        plot_atoms(x, y, frame0['z_plane'], frame0['element'], r, ax=ax_left, xlabel='x', ylabel='y', visualization_cutoff=visualization_cutoff, atom_indices=atom_indices, bonded_indices=bonded_indices)
        ax_left.text(0.98, 0.98, ha='right', va='top', transform=ax_left.transAxes,
                    s=f'Water |z| ≤ {z_threshold} Å of Cl', fontsize=12)

        if args.panels == 'all':
            pcm_xz = ax_right_top.pcolormesh(xe_xz, ze_xz, h_xz.T, cmap='Blues', shading='auto', vmin=vmin, vmax=vmax)
            plot_atoms(x, z, frame0['y_plane'], frame0['element'], r, ax=ax_right_top, xlabel='x', ylabel='z', visualization_cutoff=visualization_cutoff, atom_indices=atom_indices, bonded_indices=bonded_indices)
            ax_right_top.tick_params(axis='both', labelsize=8)

            pcm_yz = ax_right_bottom.pcolormesh(ye_yz, ze_yz, h_yz.T, cmap='Blues', shading='auto', vmin=vmin, vmax=vmax)
            plot_atoms(y, z, frame0['x_plane'], frame0['element'], r, ax=ax_right_bottom, xlabel='y', ylabel='z', visualization_cutoff=visualization_cutoff, atom_indices=atom_indices, bonded_indices=bonded_indices)
            ax_right_bottom.tick_params(axis='both', labelsize=8)

        sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        if args.panels == 'all':
            cbar = plt.colorbar(sm, ax=[ax_left, ax_right_top, ax_right_bottom], location='top', shrink=0.8, pad=0.02)
        else:
            cbar = plt.colorbar(sm, ax=ax_left, location='right', shrink=0.8, pad=0.02)
        cbar.set_label('Counts')

    plt.savefig(f'water_density_around_{selection.replace(" ", "_")}_{idx}_{density_mode}.png', dpi=300, bbox_inches='tight')
    plt.show()