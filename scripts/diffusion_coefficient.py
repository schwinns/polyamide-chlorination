# Script to calculate the diffusion coefficient from MD simulation

import numpy as np
import matplotlib.pyplot as plt

import MDAnalysis as mda
import MDAnalysis.transformations as trans
import MDAnalysis.analysis.msd as mdaMSD

import statsmodels.api as sm
from scipy import stats
from scipy.optimize import curve_fit
from tqdm import tqdm

class DiffusionCoefficient:

    def __init__(self, top, traj, membrane='resname PA*', water='resname SOL', cation='resname NA', anion='resname CL', transform=True):
        '''
        Class to calculate the diffusion coefficient from a trajectory.

        Parameters
        ----------
        top : str, None
            Name of the topology file (e.g. gro, tpr, pdb) If None, you must create a Universe manually.
        traj : str, None
            Name of the trajectory file (e.g. xtc, trr, dcd). If None, you must create a Universe manually.
        membrane : str
            MDAnalysis selection language for the membrane, default='resname PA*'
        water : str
            MDAnalysis selection language for water, default='resname SOL'
        cation : str
            MDAnalysis selection language for the cation, default='resname NA'
        anion : str
            MDAnalysis selection language for the anion, default='resname CL'
        transform : bool
            Whether to apply trajectory transformations (centering, unwrapping, nojump), default=True
        
        '''

        if top is not None or traj is not None:
            # Load in underlying MDAnalysis Universe and pre-select some atom groups
            self.universe = mda.Universe(top, traj)
            self.PA = self.universe.select_atoms(membrane)
            self.water = self.universe.select_atoms(water)
            self.cation = self.universe.select_atoms(cation)
            self.anion = self.universe.select_atoms(anion)

            # create a workflow for on-the-fly transformations
            workflow = []
            
            if transform:

                # if membrane is present, center around membrane
                if len(self.PA) > 0:
                    workflow.append(trans.unwrap(self.universe.atoms))
                    workflow.append(trans.center_in_box(self.PA, center='mass'))
                    workflow.append(trans.wrap(self.universe.atoms))

                # no jump trajectory unwrapping, necessary for diffusion coefficient calculations
                workflow.append(trans.nojump.NoJump())

                # add transformations
                self.universe.trajectory.add_transformations(*workflow)
        

    def run(self, atom_group, n_bootstraps=0, confidence=0.95, msd_kwargs={}, fit_kwargs={}):
        '''
        Run the calculation to get the MSD and perform the linear fit.

        Parameters
        ----------
        atom_group : str or MDAnalysis.AtomGroup
            Atoms to calculate the diffusion coefficient for
        n_bootstraps : int
            Number of bootstraps to perform to calculate error in the diffusion coefficient, default=0
        confidence : float
            Confidence interval for bootstrapping, should be [0,1), default=0.95
        msd_kwargs : dict
            Keyword arguments for the DiffusionCoefficient.msd method. 
            Options are 'msd_type', 'step'. Default={}
        fit_kwargs : dict
            Keyword arguments for the DiffusionCoefficient.fit_msd method. 
            Options are 'msds', 'start_idx', 'end_idx'. Default={}

        Returns
        -------
        D : float
            Diffusion coefficient for the selection
        ci : float
            Confidence interval estimate if bootstrapping is performed

        '''

        if isinstance(atom_group, str):
            ag = self.universe.select_atoms(atom_group)
            self.selection = ag
        else:
            ag = atom_group
            self.selection = ag

        # Calculate MSD and fit to get diffusion coefficient
        msd_ts, lagtimes = self.msd(ag, **msd_kwargs)
        self.fit_msd(**fit_kwargs)

        # boostrap to get a confidence interval
        if n_bootstraps > 0:
            self.msd_boots = np.zeros((msd_ts.shape[0], n_bootstraps))
            diff_boots = np.zeros(n_bootstraps)

            # bootstrap over randomly chosen particles with replacement
            for b in tqdm(range(n_bootstraps)):
                rng = np.random.default_rng()
                indices = rng.integers(0, len(ag.residues), len(ag.residues)) 
                self.msd_boots[:,b] = self.msds_by_particle[:,indices].mean(axis=1)
                diff_boots[b],_ = self.fit_msd(msds=self.msd_boots[:,b], **fit_kwargs)

            # calculate mean, standard error, and confidence interval for diffusion coefficient
            if n_bootstraps <= 30: # use t-distribution
                self.ci = stats.t.interval(confidence, n_bootstraps, loc=diff_boots.mean(), scale=stats.sem(diff_boots))
                self.msd_ci = stats.t.interval(confidence, n_bootstraps, loc=self.msd_boots.mean(axis=1), scale=stats.sem(self.msd_boots, axis=1))
            else: # use normal distribution
                self.ci = stats.norm.interval(confidence, loc=diff_boots.mean(), scale=stats.sem(diff_boots))
                self.msd_ci = stats.norm.interval(confidence, loc=self.msd_boots.mean(axis=1), scale=stats.sem(self.msd_boots, axis=1))

            self.msd_ts = self.msd_boots.mean(axis=1)
            self.msd_stderr = stats.sem(self.msd_boots, axis=1)
            self.D = diff_boots.mean()
            self.stderr = stats.sem(diff_boots)
            
            return self.D, self.ci
        
        else:
            return self.D


    def msd(self, atom_group, msd_type='xyz', step=1):
        '''
        Use MDAnalysis EinsteinMSD to calculate the MSD for a given group.

        Parameters
        ----------
        atom_group : MDAnalysis.AtomGroup
            Atom group to calculate MSD for
        msd_type : str
            Which dimensions to calculate the MSD for, default='xyz'
        step : int
            Trajectory step for which to run the MSD calculation, default=1

        Returns
        -------
        msd_ts : np.array
            Time series with the MSD averaged over all the particles with respect to lag-time
        lagtimes : np.array
            Lag-time in nanoseconds

        '''

        self.selection = atom_group
        MSD = mdaMSD.EinsteinMSD(atom_group, msd_type=msd_type, fft=True)
        MSD.run(step=step)
    
        self.msds_by_particle = MSD.results.msds_by_particle / 10**2 # save the per particle MSDs for bootstrapping, converted to nm^2
        self.msd_ts = MSD.results.timeseries / 10**2 # convert to nm^2
        dt = self.universe.trajectory[step].time - self.universe.trajectory[0].time
        self.lagtimes = np.arange(MSD.n_frames)*dt / 1000 # convert to ns
        self.dimensionality = MSD.dim_fac
        self.msd_ci = (np.zeros(MSD.n_frames), np.zeros(MSD.n_frames))
    
        return self.msd_ts, self.lagtimes
    

    def fit_msd(self, msds=None, start_idx=-1, end_idx=-1):
        '''
        Fit the MSD vs time lags to a line to extract the diffusion coefficient.

        Parameters
        ----------
        msds : np.array
            Mean squared displacement over the time lags for the linear fit, default=None means use
            previously calculated MSD saved in the object
        start_idx : int
            Starting index for the region to fit the MSD, default=-1 means start at 10% of the time lags,
            which is the same as Gromacs gmx msd defaults
        end_idx : int
            Ending index for the region to fit the MSD, default=-1 means end at 90% of the time lags,
            which is the same as Gromacs gmx msd defaults

        Returns
        -------
        D : float
            Diffusion coefficient for the calculated MSD (nm^2 / ns = D*1e-9 m^2 / s)
        stderr : float
            Standard error for D in the linear fit

        '''

        if msds is None:
            msds = self.msd_ts

        # get the region of the MSD to fit
        if start_idx == -1:
            val = np.quantile(self.lagtimes, 0.1)
            start_idx = np.where(self.lagtimes >= val)[0][0]
        if end_idx == -1:
            val = np.quantile(self.lagtimes, 0.9)
            end_idx = np.where(self.lagtimes <= val)[0][-1]

        # set up the matrices for the linear regression
        X = self.lagtimes[start_idx:end_idx]
        Y = msds[start_idx:end_idx]
        X = sm.add_constant(X)

        # fit the ordinary least squares model and get the slope
        ols = sm.OLS(Y, X)
        self.fit_results = ols.fit()
        b, m = self.fit_results.params
        be, me = self.fit_results.bse # standard error in parameters

        self.D = m / 2 / self.dimensionality
        self.stderr = me / 2 / self.dimensionality

        return self.D, self.stderr # nm^2 / ns = 1e-9 m^2 / s
    

    def restrict_to_membrane(self, atom_group, frac=0.8, membrane_bounds=None, membrane_fraction=None, d=2):
        '''
        Restrict the MSD calculations to only the atoms in the selection that remain inside
        the membrane for a desired fraction of the trajectory.

        Parameters
        ----------
        atom_group : str or MDAnalysis.AtomGroup
            Atoms to calculate the MSD for
        frac : float
            Fraction of the trajectory that the selection must remain inside the membrane to be saved. Should be in (0,1], default=0.8
        membrane_bounds : array-like
            The bounds of the membrane in Angstroms, default=None means the full membrane coordinate
        membrane_fraction : array-like
            The bounds of the membrane in percentage of the membrane, default=None means the full membrane coordinate
        d : int
            Dimension to consider, default=2 means the z dimension

        Returns
        -------
        frac_inside : np.array
            Array containing the fraction of time each atom in the selection is inside the membrane
        inside_membrane : MDAnalysis.AtomGroup
            The subset of the input `atom_group` that remains in the membrane

        '''

        # parse inputs
        if isinstance(atom_group, str):
            ag = self.universe.select_atoms(atom_group)
        else:
            ag = atom_group

        if membrane_bounds is None and membrane_fraction is None:
            q_lb = 0
            q_ub = 1
        elif membrane_bounds is not None:
            lb = membrane_bounds[0]
            ub = membrane_bounds[1]
        elif membrane_fraction is not None:
            q_lb = membrane_fraction[0]
            q_ub = membrane_fraction[1]

        dims = {0 : 'x', 1 : 'y', 2 : 'z'}
        dim = dims[d]

        # track which atoms are within the membrane for each frame
        print('Tracking which atoms stay within the membrane...')
        is_inside = np.zeros((len(self.universe.trajectory), len(ag)), dtype=np.int8)
        for i,ts in tqdm(enumerate(self.universe.trajectory)):
            
            # get the bounds for this frame, if membrane_bounds is not specified
            if membrane_bounds is None:
                lb = np.quantile(self.PA.positions[:,d], q_lb)
                ub = np.quantile(self.PA.positions[:,d], q_ub)

            # get indices of atoms above lower bound
            is_inside[i,:] += ag.positions[:,d] >= lb # 1 if T, 0 if F

            # get indices of atoms below upper bound
            is_inside[i,:] += ag.positions[:,d] <= ub # 2 if TT, 1 if TF/FT, 0 if FF 

            # coerce indices to be 1 if both above lb and below ub and 0 otherwise
            is_inside[i,:] = is_inside[i,:] // 2

        # convert the tracking array into a fraction of time spent in membrane
        self.frac_inside = is_inside.sum(axis=0) / len(self.universe.trajectory)

        return self.frac_inside, ag[self.frac_inside >= frac]
    

    def _power_law_func(self, t, A, alpha):
        '''Power law function to use with fitting'''
        return A*np.power(t, alpha)
    

    def fit_power_law(self, initial_guess=[1,1], n_bootstraps=0, confidence=0.95, start_idx=0, end_idx=None):
        '''
        Fit power law to the MSD curve. MSD = A*t^alpha, where alpha = 1 is normal diffusion, alpha < 1 is 
        subdiffusion, and alpha > 1 is superdiffusion. 

        Parameters
        ----------
        initial_guess : array-like
            Initial guess for the power law parameters, A and alpha, default=[1,1]
        n_bootstraps : int
            Number of bootstraps to calculate error in the power law parameters, default=0
        confidence : float
            Confidence interval for bootstrapping, should be [0,1), default=0.95
        start_idx : int
            Starting index for the region to fit the MSD, default=0
        end_idx : int
            Ending index for the region to fit the MSD, default=None means end at 50% of the time lags

        Returns
        -------
        params : np.array
            Fitted power law parameters [A, alpha]
        params_ci : np.array
            If bootstrapped error, confidence intervals for the fitted power law parameters [[A_lb,     A_ub],
                                                                                             [alpha_lb, alpha_ub]]

        '''

        # parse inputs
        if end_idx is None:
            end_idx = int(self.lagtimes.shape[0]/2)

        # no bootstrapping
        if n_bootstraps == 0:
            params, covariance = curve_fit(self._power_law_func, self.lagtimes[start_idx:end_idx], self.msd_ts[start_idx:end_idx], p0=initial_guess)
            self.A = params[0]
            self.alpha = params[1]

            return params
        
        # bootstrap to get confidence interval
        else:
            self.msd_boots = np.zeros((self.msd_ts.shape[0], n_bootstraps))
            param_boots = np.zeros((2,n_bootstraps))
            self.power_law_boots = np.zeros((self.msd_ts.shape[0], n_bootstraps))

            # bootstrap over randomly chosen particles with replacement
            for b in tqdm(range(n_bootstraps)):
                rng = np.random.default_rng()
                indices = rng.integers(0, len(self.selection.residues), len(self.selection.residues)) 
                self.msd_boots[:,b] = self.msds_by_particle[:,indices].mean(axis=1)
                param_boots[:,b],_ = curve_fit(self._power_law_func, self.lagtimes[start_idx:end_idx], self.msd_boots[start_idx:end_idx,b], p0=initial_guess)
                self.power_law_boots[:,b] = self._power_law_func(self.lagtimes[start_idx:end_idx], param_boots[0,b], param_boots[1,b]) # save fits for error in plot

            # calculate mean, standard error, and confidence interval for diffusion coefficient
            if n_bootstraps <= 30: # use t-distribution
                self.A_ci = stats.t.interval(confidence, n_bootstraps, loc=param_boots[0,:].mean(), scale=stats.sem(param_boots[0,:]))
                self.alpha_ci = stats.t.interval(confidence, n_bootstraps, loc=param_boots[1,:].mean(), scale=stats.sem(param_boots[1,:]))
            else: # use normal distribution
                self.A_ci = stats.norm.interval(confidence, loc=param_boots[0,:].mean(), scale=stats.sem(param_boots[0,:]))
                self.alpha_ci = stats.norm.interval(confidence, loc=param_boots[1,:].mean(), scale=stats.sem(param_boots[1,:]))

            self.msd_ts = self.msd_boots.mean(axis=1)
            self.msd_stderr = stats.sem(self.msd_boots, axis=1)
            self.A = param_boots[0,:].mean()
            self.alpha = param_boots[1,:].mean()

            params = param_boots.mean(axis=1)
            params_ci = np.vstack((self.A_ci, self.alpha_ci))

            return params, params_ci
    

    def plot_msd(self, fraction=0.5, filename='msd.dat', ax=None):
        '''
        Plot the MSD versus the time lag.

        Parameters
        ----------
        fraction : float
            Fraction of the MSD curve to show, should be between [0,1], default=0.5
        filename : str
            Filename for saving the data, default='msd.dat'
        ax : matplotlib.axes.Axes
            Axes to plot the MSD on, default=None means create new axes

        Returns
        -------
        ax : matplotlib.axes.Axes
            Plot of the MSD

        '''

        end_idx = int(fraction * self.lagtimes.shape[0])

        if ax is None:
            fig, ax = plt.subplots(1,1)
        
        ax.plot(self.lagtimes[:end_idx], self.msd_ts[:end_idx])
        ax.fill_between(self.lagtimes[:end_idx], 
                        self.msd_ci[0][:end_idx], 
                        self.msd_ci[1][:end_idx], alpha=0.5)

        ax.set_xlabel('time lag (ns)')
        ax.set_ylabel('MSD (nm$^2$)')

        if filename is not None:
            np.savetxt(filename, np.vstack([self.lagtimes, self.msd_ts, self.msd_ci[0], self.msd_ci[1]]).T,
                       header='lag times (ns), MSD (nm^2), lower confidence bound (nm^2), upper confidence bound (nm^2)')
            
        return ax
    

    def plot_power_law(self, start_idx=0, end_idx=None):
        '''
        Plot the power law fit with the MSD data.

        Parameters
        ----------
        start_idx : int
            Starting index for the region to fit the MSD, default=0
        end_idx : int
            Ending index for the region to fit the MSD, default=None means end at 50% of the time lags

        Returns
        -------
        ax : matplotlib.axes.Axes

        '''

        # parse inputs
        if end_idx is None:
            end_idx = int(self.lagtimes.shape[0]/2)

        # get the fitted power law curve
        fit = self.power_law_boots.mean(axis=1)
        fit_stderr = stats.sem(self.power_law_boots, axis=1)

        fig, ax = plt.subplots(1,1)
        
        ax.plot(self.lagtimes[start_idx:end_idx], self.msd_ts[start_idx:end_idx], label='MSD')
        ax.plot(self.lagtimes[start_idx:end_idx], fit, ls='dashed', c='black', label='power law fit')
        ax.fill_between(self.lagtimes[start_idx:end_idx], 
                        self.msd_ts[start_idx:end_idx]-self.msd_ci[0][start_idx:end_idx], 
                        self.msd_ts[start_idx:end_idx]+self.msd_ci[1][start_idx:end_idx], alpha=0.5)
        ax.fill_between(self.lagtimes[start_idx:end_idx], 
                        fit-fit_stderr, 
                        fit+fit_stderr, 
                        alpha=0.5, facecolor='black')

        ax.set_xlabel('time lag (ns)')
        ax.set_ylabel('MSD (nm$^2$)')

        return ax



    



        

        
