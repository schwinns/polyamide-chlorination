# Pull MSDs from DiffusionCoefficient objects and report diffusion coefficients with confidence intervals for waters and ions, both inside and outside the membrane. Also report the MSD plots with confidence intervals and standard errors.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy import stats
import statsmodels.api as sm

import MDAnalysis as mda

def load_pickle(filename):
    '''Load pickle file with a Python object'''
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

def bootstrap_water_diffusion_coefficients(diff, n_bootstraps=1000, start_pct=0.1, end_pct=0.5, plot=False):
    '''Bootstrap water diffusion coefficients from MSDs'''

    n_lagtimes = len(diff.lagtimes)
    n_waters = diff.msds_by_particle.shape[1]

    val = np.quantile(diff.lagtimes, start_pct)
    start_idx = np.where(diff.lagtimes >= val)[0][0]
    val = np.quantile(diff.lagtimes, end_pct)
    end_idx = np.where(diff.lagtimes <= val)[0][-1]

    X = diff.lagtimes[start_idx:end_idx]
    X = sm.add_constant(X)
    print('Fitting lagtimes between {:.1f} and {:.1f} ns'.format(diff.lagtimes[start_idx], diff.lagtimes[end_idx]))

    msd_boots = np.zeros((n_lagtimes, n_bootstraps))
    diff_boots = np.zeros(n_bootstraps)
    for b in range(n_bootstraps):
        rng = np.random.default_rng()
        indices = rng.choice(n_waters, size=n_waters, replace=True)
        msd_boots[:, b] = diff.msds_by_particle[:, indices].mean(axis=1)
        Y_boot = msd_boots[:, b][start_idx:end_idx]
        ols_boot = sm.OLS(Y_boot, X)
        res_boot = ols_boot.fit()
        diff_boots[b] = res_boot.params[1] / 2 / diff.dimensionality

    msd_ci = stats.norm.interval(0.95, loc=msd_boots.mean(axis=1), scale=stats.sem(msd_boots, axis=1))
    diff_ci = stats.norm.interval(0.95, loc=diff_boots.mean(), scale=stats.sem(diff_boots))
    print(f'Diffusion coefficient: {diff_boots.mean():.8f} nm^2/ns, err: {diff_boots.mean() - diff_ci[0]:.8f}')
    print(f'95% CI: [{diff_ci[0]:.8f}, {diff_ci[1]:.8f}]')

    if plot:
        fig, ax = plt.subplots(1,1, figsize=(8,6))

        # plot MSD of all waters
        for i in range(n_waters-1):
            plt.plot(diff.lagtimes, diff.msds_by_particle[:, i], color='k', alpha=0.1)

        plt.plot(diff.lagtimes, diff.msds_by_particle[:, -1], label='individual water', color='k', alpha=0.1)

        # plot the bootstrapped mean and confidence intervals
        plt.plot(diff.lagtimes, msd_boots.mean(axis=1), label='bootstrapped mean', color='k')
        plt.plot(diff.lagtimes, msd_ci[0], label='95% CI', color='g', ls='--')
        plt.plot(diff.lagtimes, msd_ci[1], color='g', ls='--')

        # plot the fitting range
        plt.axvline(diff.lagtimes[start_idx], color='r', ls='--')
        plt.axvline(diff.lagtimes[end_idx], color='r', ls='--')

        # formatting
        plt.xlabel('Lag time (ns)')
        plt.ylabel('Mean squared displacement (nm$^2$)')
        plt.legend(loc='upper left')
        plt.xlim(0, 1000)
        plt.ylim(0, 2000)

    return diff_boots.mean(), diff_ci


def bootstrap_ion_diffusion_coefficients(diff, idx, n_bootstraps=1000, start_pct=0.1, end_pct=0.5, plot=False):
    '''Bootstrap ion diffusion coefficients from MSDs'''

    n_lagtimes = len(diff.lagtimes)
    n_ions = len(idx)

    val = np.quantile(diff.lagtimes, start_pct)
    start_idx = np.where(diff.lagtimes >= val)[0][0]
    val = np.quantile(diff.lagtimes, end_pct)
    end_idx = np.where(diff.lagtimes <= val)[0][-1]

    X = diff.lagtimes[start_idx:end_idx]
    X = sm.add_constant(X)
    print('Fitting lagtimes between {:.1f} and {:.1f} ns'.format(diff.lagtimes[start_idx], diff.lagtimes[end_idx]))

    msd_boots = np.zeros((n_lagtimes, n_bootstraps))
    diff_boots = np.zeros(n_bootstraps)
    for b in range(n_bootstraps):
        rng = np.random.default_rng()
        indices = rng.choice(idx, size=n_ions, replace=True)
        msd_boots[:, b] = diff.msds_by_particle[:, indices].mean(axis=1)
        Y_boot = msd_boots[:, b][start_idx:end_idx]
        ols_boot = sm.OLS(Y_boot, X)
        res_boot = ols_boot.fit()
        diff_boots[b] = res_boot.params[1] / 2 / diff.dimensionality

    msd_ci = stats.norm.interval(0.95, loc=msd_boots.mean(axis=1), scale=stats.sem(msd_boots, axis=1))
    diff_ci = stats.norm.interval(0.95, loc=diff_boots.mean(), scale=stats.sem(diff_boots))
    print(f'Diffusion coefficient: {diff_boots.mean():.8f} nm^2/ns, err: {diff_boots.mean() - diff_ci[0]:.8f}')
    print(f'95% CI: [{diff_ci[0]:.8f}, {diff_ci[1]:.8f}]')

    if plot:
        fig, ax = plt.subplots(1,1, figsize=(8,6))

        # plot MSD of all waters
        for i in idx[:-1]:
            plt.plot(diff.lagtimes, diff.msds_by_particle[:, i], color='k', alpha=0.1)

        plt.plot(diff.lagtimes, diff.msds_by_particle[:, idx[-1]], label='individual water', color='k', alpha=0.1)

        # plot the bootstrapped mean and confidence intervals
        plt.plot(diff.lagtimes, msd_boots.mean(axis=1), label='bootstrapped mean', color='k')
        plt.plot(diff.lagtimes, msd_ci[0], label='95% CI', color='g', ls='--')
        plt.plot(diff.lagtimes, msd_ci[1], color='g', ls='--')

        # plot the fitting range
        plt.axvline(diff.lagtimes[start_idx], color='r', ls='--')
        plt.axvline(diff.lagtimes[end_idx], color='r', ls='--')

        # formatting
        plt.xlabel('Lag time (ns)')
        plt.ylabel('Mean squared displacement (nm$^2$)')
        plt.legend(loc='upper left')
        plt.xlim(0, 1000)
        plt.ylim(0, 20)

    return diff_boots.mean(), diff_ci


if __name__ == '__main__':

    label = 'Pristine'
    start_pct = 0.05
    end_pct = 0.25

    # Load water diffusion coefficient objects
    diff = load_pickle('diffusion_coefficient_unrestricted.pkl')

    # Get bootstrapped diffusion coefficients for waters    
    print('\nDiffusion coefficients for all waters:')
    D_unrestricted, CI_unrestricted = bootstrap_water_diffusion_coefficients(diff, 
                                                                             n_bootstraps=1000, 
                                                                             start_pct=start_pct, 
                                                                             end_pct=end_pct,
                                                                             plot=False)
    
    # Load ion diffusion coefficient object and MDAnalysis universe
    diff_ion = load_pickle('ion_diffusion_coefficient.pkl')

    # Get bootstrapped diffusion coefficients for ions    
    print('\nDiffusion coefficients for all ions:')
    D_ion_unrestricted, CI_ion_unrestricted = bootstrap_ion_diffusion_coefficients(diff_ion, 
                                                                         np.arange(len(diff_ion.cation)), 
                                                                         n_bootstraps=1000, 
                                                                         start_pct=start_pct, 
                                                                         end_pct=end_pct,
                                                                         plot=False)
    
    plt.show()