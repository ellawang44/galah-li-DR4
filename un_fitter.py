import numpy as np
import ultranest
from scipy.stats import norm


class UNFitter():

    def __init__(self, wl_obs, flux_obs, flux_err, fitter, constraints, mode, metal_poor, opt=None, nwalkers=8, run=True, grid=None, fit_rv=True):
        self.wl_obs = wl_obs
        self.flux_obs = flux_obs
        self.flux_err = flux_err
        self.fitter = fitter
        self.grid = grid
        self.c = 299792.458
        # define variables
        self.constraints = constraints
        ndim = len(self.constraints)
        self.mode = mode
        self.metal_poor = metal_poor
        # randomise initial position of walkers
        p0 = np.zeros((nwalkers, ndim))
        for i in range(nwalkers):
            for j in range(ndim):
                minP, maxP = constraints[j]
                p0[i][j] = np.random.uniform(minP, maxP)
        # names of columns
        if metal_poor:
            if fit_rv:
                param_names = ['Li', 'rv', 'const']
            else:
                param_names = ['Li', 'const']
        else:
            param_names = ['Li', 'CN1', 'Fe', 'CN2', 'Ce/V', '?1', '?2', 'const']
        # if run the sample, or else just initialize class
        if run:
            self.sampler = ultranest.ReactiveNestedSampler(param_names, self.like, self.transform)
            self.results = self.sampler.run(viz_callback=False, show_status=False)

    def prob(self, param):
        '''
        param: the parameters of the current iteration
        combines the prior with the likelihood function
        '''

        prior = self.prior(param)
        if not np.isfinite(prior):
            return -np.inf
        return self.prior(param) + self.like(param)

    def prior(self, param):
        '''
        param: the parameters of the current iteration
        prior function, uniform with limits currently
        '''

        for i, p in enumerate(param):
            if (p < self.constraints[i][0]) or (self.constraints[i][1] < p):
                return -np.inf
        prior = 1
        return np.log(prior)

    def transform(self, param):
        '''from 0-1 to original parameter space
        '''

        new_params = []
        for i, p in enumerate(param):
            scale = self.constraints[i][1] - self.constraints[i][0]
            new_params.append((p * scale) + self.constraints[i][0])
        return new_params

    def like(self, param):
        '''
        param: the parameters of the current iteration
        The likelihood function, logged so you can sum
        '''
        if self.metal_poor:
            rv = 0
            std = self.fitter.std
        else:
            rv = self.fitter.rv
            std = self.fitter.std
        wl_left = 6706.730*(1+rv/self.c)-std*2
        wl_right = 6708.961*(1+rv/self.c)+std*2
        mask = (wl_left <= self.wl_obs) & (self.wl_obs <= wl_right)
        return -np.sum(np.square((self.fitter.model(self.wl_obs[mask], param, grid=self.grid) - self.flux_obs[mask])/self.flux_err[mask]))

