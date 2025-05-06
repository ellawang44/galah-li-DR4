from fit import FitBroad, FitG, FitGFixed, FitB, FitBFixed, FitVFixed, iter_fit, amp_to_init, pred_amp, cc_rv, _wl, bline, gline, chisq
from synth import _spectra, Grid, _c
import numpy as np
import matplotlib.pyplot as plt
from breidablik.interpolate.spectra import Spectra
from breidablik.analysis import read, tools
from config import *
from scipy.interpolate import interp1d, CubicSpline
from astro_tools import vac_to_air
from un_fitter import UNFitter
import corner
import time
from stats import Stats

class FitSpec:
    '''Fitting 1 spectrum, contains plotting, and save/load.
    '''

    def __init__(self, std, snr, sid, teff, logg, feh, rv_lim=None):
        '''
        Parameters
        ----------
        std : float
            std from galah, in \AA.
        snr : float
            The SNR per pixel of the spectrum.
        sid : int
            The sobject_id of the spectrum.
        teff : float
            The GALAH DR# of teff for the spectrum
        logg : float
            The GALAH DR# of logg for the spectrum
        feh : float
            The GALAH DR# of feh for the spectrum
        rv_lim : float, optional
            The limit on rv, mirrored limit on either side, it is based on the galah vbroad value, in km/s.
        '''

        self.std = std
        self.snr = snr
        self.sid = sid
        self.teff = teff
        self.logg = logg
        self.feh = feh
        self.rv_lim = rv_lim # don't need this parameter if you aren't fitting
        self.li_center = 6707.8139458 # weighted mean of smith98 Li7 line centers
        if self.grid_check(np.array([teff]), np.array([logg]), np.array([feh]))[0]: # valid sp from galah
            self.mode = 'Breidablik'
        else:
            self.mode = 'Gaussian'
        # make model to translate between rew and abundance
        # can't run breidablik with nans
        if self.mode == 'Breidablik':
            self.gen_ew_to_abund()

    def grid_check(self, teffs, loggs, fehs):
        '''Check that this combination of sp is within the Breidablik grid.

        Parameters
        ----------
        teff : float
            Effective temperature
        logg : float
            Surface gravity
        feh : float
            Metallicity

        Returns
        -------
        in_grid : bool
            True if the stellar parameters are in the Breidablik grid. False otherwise.
        '''

        with open('grid_snapshot.txt', 'r') as f:
            t_step, m_step = np.float_(f.readline().split())
            grid = np.loadtxt(f)
        scaled_sp = np.array([teffs*t_step, loggs, fehs*m_step]).T
        tile = np.array([np.tile(sp, (grid.shape[0], 1)) for sp in scaled_sp])
        dist = np.sqrt(np.sum(np.square(grid - tile), axis = 2))
        min_dist = np.min(dist, axis=1)
        in_grid = np.sqrt(3*0.25**2) > min_dist
        return in_grid

    def gen_ew_to_abund(self):
        '''Generate the function converting rew to abundances
        '''

        # calculate corresponding ews
        abunds = list(np.arange(-0.5, 5.1, 0.18)) # extrapolating 1 dex
        ews = np.array([self.li_center*10**tools.rew(_wl, spec, center=6709.659, upper=100, lower=100) for spec in _spectra._predict_flux(self.teff, self.logg, self.feh, abunds)])
        # set values
        self.min_ew = ews[0]
        self.max_ew = ews[-1]
        self.ew_to_abund = lambda x: CubicSpline(ews, abunds)(x)

    def fit_broad(self, spectra, center=np.array([6696.085, 6698.673, 6703.565, 6705.101, 6710.317, 6711.819, 6713.095, 6713.742, 6717.681])):
        '''Fit the broad region of the spectrum, ews, std simultaneously. None if the star is poorly constrained (less than 3 lines with amplitudes above noise).

        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm)
        center : 1darray
            The centers of the lines used in the fitting. Default values:
            np.array([6696.085, 6698.673, 6703.565, 6705.101, 6710.317, 6711.819, 6713.095, 6713.742, 6717.681])
        '''

        res = iter_fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], center=center, std=self.std, rv_lim=self.rv_lim)
        if res is None: # poorly constrained
            self.poorly_constrained = True
            self.broad_fit = None
        else: # well constrained
            self.poorly_constrained = False
            self.broad_fit = res
        # save the centers used
        self.broad_center = center

    def mp_init(self, spectra):
        '''Init values for poorly constrained fit.

        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm)

        Returns
        -------
        init : dict
            Initial values.
        '''

        amps, _, const = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], centers=[self.li_center])
        init = amp_to_init(amps, self.std, const)
        return init

    def fit_li(self, spectra, center=np.array([6707.8139458, 6706.730, 6707.433, 6707.545, 6708.096, 6708.810, 6709.011])):
        '''Fit the Li region of the spectrum, fits Li only if poorly constrained (less than 3 lines with amplitudes above noise), or else fits all blending lines too.

        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm)
        center : 1darray
            The centers of the lines used in the fitting. First value is Li, not really used in the fitting, but needed for the initial guess. Default values:
            np.array([6707.8139458, 6706.730, 6707.433, 6707.545, 6708.096, 6708.810, 6709.01]))
        '''

        self.narrow_center = center # save centers used

        # TODO: testing voigt function, the mode should be set in the init function
        if self.mode == 'Voigt':
            self.fit_voigt(spectra)
            return None

        # if any sp is nan, then fit with Gaussian, because Breidablik breaks
        if self.mode == 'Gaussian':
            self.fit_gaussian(spectra)
            return None

        if self.poorly_constrained:
            # if metal poor, no CN, because otherwise it's uncontrained again
            fitter = FitBFixed([self.li_center], self.std, 0, self.teff, self.logg, self.feh, self.ew_to_abund, self.min_ew, max_ew=self.max_ew)
            init = self.mp_init(spectra)
            # fit
            res = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=init)
            # turn results into consistent format
            res['amps'] = [0]*len(self.narrow_center[1:])
        else:
            fitter = FitBFixed(self.narrow_center[1:], self.std, self.broad_fit['rv'], self.teff, self.logg, self.feh, self.ew_to_abund, self.min_ew, max_ew=self.max_ew)
            # initial guess from the amplitudes in the spectrum (a little bit overestimated)
            amps, _, const = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], centers=self.narrow_center, rv=self.broad_fit['rv'])
            init = amp_to_init(amps, self.std, const, rv=self.broad_fit['rv'])
            init['amps'][0] = min(max(init['amps'][0], self.min_ew), self.max_ew)
            # calculate ratio
            pred = fitter.model(spectra['wave_norm'], fitter.get_init(init))
            pred_amps, _, const = pred_amp(spectra['wave_norm'], pred, spectra['uob_norm'], centers=self.narrow_center, rv=self.broad_fit['rv'])
            ratio = pred_amps/amps
            ratio[np.isnan(ratio)] = 1 # sometimes there's a divide by 0 where amps is 0
            # better amps
            amps = amps/ratio
            init = amp_to_init(amps, self.std, const, rv=self.broad_fit['rv'])
            init['amps'][0] = min(max(init['amps'][0], self.min_ew), self.max_ew)
            # fit
            res = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=init)
        # save Li fit
        self.li_init_fit = {'amps':[res['li'], *res['amps']], 'std':res['std_li'], 'rv':res['rv'], 'const':res['const'], 'minchisq':res['minchisq']}

    def fit_gaussian(self, spectra, center=np.array([6707.8139458, 6706.730, 6707.433, 6707.545, 6708.096, 6708.810, 6709.011])):
        '''Fit the Li region of the spectrum, fits only Li line if poorly constrained (less than 3 lines with amplitudes above noise), or else fits blending lines as well.

        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm)
        center : 1darray
            The centers of the lines used in the fitting. First value is Li, not really used in the fitting, but needed for the initial guess. Default values:
            np.array([6707.8139458, 6706.730, 6707.433, 6707.545, 6708.096, 6708.810, 6709.011]))
        '''

        if self.poorly_constrained:
            # if metal poor, no CN, because otherwise it's uncontrained again
            fitter = FitGFixed(center=[self.li_center], std=self.std, rv=0)
            init = self.mp_init(spectra)
            # fit
            res  = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=init)
            # turn results into consistent format
            res['amps'] = [0]*len(self.narrow_center[1:])
        else:
            fitter = FitGFixed(self.narrow_center, self.std, self.broad_fit['rv'])
            # initial guess from the amplitudes in the spectrum (a little bit overestimated)
            amps, _, const = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], centers=self.narrow_center, rv=self.broad_fit['rv'])
            init = amp_to_init(amps, self.std, const, rv=self.broad_fit['rv'])
            # calculate ratio
            pred = fitter.model(spectra['wave_norm'], fitter.get_init(init))
            pred_amps, _, const = pred_amp(spectra['wave_norm'], pred, spectra['uob_norm'], centers=self.narrow_center, rv=self.broad_fit['rv'])
            ratio = pred_amps/amps
            ratio[np.isnan(ratio)] = 1 # sometimes there's a divide by 0 where amps is 0
            # better amps
            amps = amps/ratio
            init = amp_to_init(amps, self.std, const, rv=self.broad_fit['rv'])
            # fit
            res = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=init)
        # save initial Li fit
        self.li_init_fit = {'amps':[res['li'], *res['amps']], 'const':res['const'], 'std':res['std_li'], 'rv':res['rv'], 'minchisq':res['minchisq']}

    def fit_voigt(self, spectra, center=np.array([6707.8139458, 6706.730, 6707.433, 6707.545, 6708.096, 6708.810, 6709.011])):
        '''Fit the Li region of the spectrum, fits only Li line if poorly constrained (less than 3 lines with amplitudes above noise), or else fits blending lines as well.

        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm)
        center : 1darray
            The centers of the lines used in the fitting. First value is Li, not really used in the fitting, but needed for the initial guess. Default values:
            np.array([6707.8139458, 6706.730, 6707.433, 6707.545, 6708.096, 6708.810, 6709.011]))
        '''

        if self.poorly_constrained:
            # if metal poor, no CN, because otherwise it's uncontrained again
            fitter = FitVFixed(center=[self.li_center], std=self.std, rv=0)
            init = self.mp_init(spectra)
            # fit
            res  = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=init)
            # turn results into consistent format
            res['amps'] = [0]*len(self.narrow_center[1:])
        else:
            fitter = FitVFixed(self.narrow_center, self.std, self.broad_fit['rv'])
            # initial guess from the amplitudes in the spectrum (a little bit overestimated)
            amps, _, const = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], centers=self.narrow_center, rv=self.broad_fit['rv'])
            init = amp_to_init(amps, self.std, const, rv=self.broad_fit['rv'])
            # calculate ratio
            pred = fitter.model(spectra['wave_norm'], fitter.get_init(init))
            pred_amps, _, const = pred_amp(spectra['wave_norm'], pred, spectra['uob_norm'], centers=self.narrow_center, rv=self.broad_fit['rv'])
            ratio = pred_amps/amps
            ratio[np.isnan(ratio)] = 1 # sometimes there's a divide by 0 where amps is 0
            # better amps
            amps = amps/ratio
            init = amp_to_init(amps, self.std, const, rv=self.broad_fit['rv'])
            # fit
            res = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=init)
        # save initial Li fit
        self.li_init_fit = {'amps':[res['li'], *res['amps']], 'const':res['const'], 'std':res['std_li'], 'rv':res['rv'], 'sigma':res['sigma'], 'gamma':res['gamma'], 'minchisq':res['minchisq']}

    def bad_spec(self, spectra):
        '''Identify bad spectra. Based on normalised flux and std being reasonable.

        Parameters
        ----------
        spectra : dict
            The dictionary containing the GALAH spectra.

        Returns
        -------
        bad : bool
            True if the spectra is bad, False otherwise.
        '''

        # below 0 or extremely above 1 spectra
        lower, upper = np.percentile(spectra['sob_norm'], [5, 95])
        if lower < 0 or upper > 1.5:
            return True
        return False

    def get_err(self):
        '''error from norris formula
        '''

        error_factor = np.sqrt(3*np.pi)/(np.pi**(1/4))
        R = 25500
        npix = 5
        self.norris = self.li_center*npix**0.5/(R*self.snr)

    def posterior_setup(self, li_factor=5, blend_factor=5, const_range=0.1, fit_rv=True):
        '''Set up the fitter, bounds, grid required to sample posterior.

        Parameters
        ----------
        li_factor : float
            The factor that the Li error is mulitplied by to create the range
        blend_factor : float
            The factor of the blends errors are mulitplied by to create the range
        const_range : float
            The amount that the continuum constant can vary by, both up and down.
        fit_rv : bool
            Controls if rv is fit or not

        Returns
        -------
        fitter : object
            The fitter object that contains the model. Different depending on mode (Breidablik or Gaussian) and poorly constrained.
        bounds : 2darray
            The boundary conditions for the walkers.
        grid : object
            For speeding up the calculations. Gaussian convolution for rotation is slow, instead we create a grid at certain abundances and vsini, then cubic spline interpolation along this grid. See synth.py
        '''
        if self.poorly_constrained and self.mode == 'Breidablik' and fit_rv:
            fitter = FitB(self.teff, self.logg, self.feh, self.std, self.ew_to_abund, self.min_ew, max_ew=self.max_ew, rv_lim=self.rv_lim, std_li=self.li_init_fit['std'])
            opt = fitter.get_init(self.li_init_fit)
            bounds = [(max(opt[0]-self.norris*li_factor, -self.max_ew), min(opt[0]+self.norris*li_factor, self.max_ew)),
                    (-self.rv_lim, self.rv_lim),
                    (opt[-1]-const_range, opt[-1]+const_range)
                    ]
        elif self.poorly_constrained and self.mode == 'Breidablik' and not fit_rv:
            fitter = FitBFixed([self.li_center], self.std, 0, self.teff, self.logg, self.feh, self.ew_to_abund, self.min_ew, max_ew=self.max_ew, std_li=self.li_init_fit['std'])
            opt = fitter.get_init(self.li_init_fit)
            bounds = [(max(opt[0]-self.norris*li_factor, -self.max_ew), min(opt[0]+self.norris*li_factor, self.max_ew)),
                    (opt[-1]-const_range, opt[-1]+const_range)
                    ]
        elif not self.poorly_constrained and self.mode == 'Breidablik':
            fitter = FitBFixed(self.narrow_center[1:], self.std, self.li_init_fit['rv'], self.teff, self.logg, self.feh, self.ew_to_abund, self.min_ew, max_ew=self.max_ew, std_li=self.li_init_fit['std'])
            opt = fitter.get_init(self.li_init_fit)
            bounds = [(max(opt[0]-self.norris*li_factor, -self.max_ew), min(opt[0]+self.norris*li_factor, self.max_ew)),
                    (max(0, opt[1]-self.norris*blend_factor), opt[1]+self.norris*blend_factor),
                    (max(0, opt[2]-self.norris*blend_factor), opt[2]+self.norris*blend_factor),
                    (max(0, opt[3]-self.norris*blend_factor), opt[3]+self.norris*blend_factor),
                    (max(0, opt[4]-self.norris*blend_factor), opt[4]+self.norris*blend_factor),
                    (max(0, opt[5]-self.norris*blend_factor), opt[5]+self.norris*blend_factor),
                    (max(0, opt[6]-self.norris*blend_factor), opt[6]+self.norris*blend_factor),
                    (opt[-1]-const_range, opt[-1]+const_range)
                    ]
        elif self.poorly_constrained and self.mode == 'Gaussian' and fit_rv:
            fitter = FitG(std=self.std, rv_lim=self.rv_lim)
            opt = fitter.get_init(self.li_init_fit)
            bounds = [(opt[0]-self.norris*li_factor, opt[0]+self.norris*li_factor),
                    (-self.rv_lim, self.rv_lim),
                    (opt[-1]-const_range, opt[-1]+const_range)
                    ]
        elif self.poorly_constrained and self.mode == 'Gaussian' and not fit_rv:
            fitter = FitGFixed([self.li_center], std=self.std, rv=0)
            opt = fitter.get_init(self.li_init_fit)
            bounds = [(opt[0]-self.norris*li_factor, opt[0]+self.norris*li_factor),
                    (opt[-1]-const_range, opt[-1]+const_range)
                    ]
        elif not self.poorly_constrained and self.mode == 'Gaussian':
            fitter = FitGFixed(self.narrow_center, self.std, rv=self.broad_fit['rv'])
            opt = fitter.get_init(self.li_init_fit)
            bounds = [(opt[0]-self.norris*li_factor, opt[0]+self.norris*li_factor),
                    (max(0, opt[1]-self.norris*blend_factor), opt[1]+self.norris*blend_factor),
                    (max(0, opt[2]-self.norris*blend_factor), opt[2]+self.norris*blend_factor),
                    (max(0, opt[3]-self.norris*blend_factor), opt[3]+self.norris*blend_factor),
                    (max(0, opt[4]-self.norris*blend_factor), opt[4]+self.norris*blend_factor),
                    (max(0, opt[5]-self.norris*blend_factor), opt[5]+self.norris*blend_factor),
                    (max(0, opt[6]-self.norris*blend_factor), opt[6]+self.norris*blend_factor),
                    (opt[-1]-const_range, opt[-1]+const_range)
                    ]

        if self.mode == 'Breidablik':
            grid = Grid(bounds[0], std=fitter.std_li, teff=self.teff, logg=self.logg, feh=self.feh, ew_to_abund=self.ew_to_abund, min_ew=self.min_ew)
        elif self.mode == 'Gaussian':
            grid = None

        return fitter, bounds, grid

    def run_post(self, spectra, fitter, bounds, grid, fit_rv=True):
        '''run ultranest, store all results in dictionary and incrememt run number.
        '''

        # run
        start = time.time()
        un_fitter = UNFitter(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], fitter, bounds, mode=self.mode, poorly_constrained=self.poorly_constrained, grid=grid, fit_rv=fit_rv)
        end = time.time()
        results = un_fitter.results
        t = end - start

        # check if on edge
        _, argmax, uniform = self.get_map(results['samples'])
        is_on_edge, due_to_const, edge_ind = self.on_edge(argmax, bounds, uniform)

        self.runs += 1
        self.run_res[self.runs] = {'results':results, 'std_li':fitter.std_li, 'time':t, 'bounds':bounds, 'posterior_good':not is_on_edge, 'due_to_const':due_to_const, 'edge_ind':edge_ind}

    def run_post_widen(self, spectra, fit_rv=True):
        '''Sample posterior, then widen if required.'''

        # set up bounds and fitters
        fitter, bounds, grid = self.posterior_setup(fit_rv=fit_rv)

        # run mcmc
        self.run_post(spectra, fitter, bounds, grid, fit_rv=fit_rv)

        # widen and rerun
        if not self.run_res[self.runs]['posterior_good']:
            if self.run_res[self.runs]['due_to_const']:
                const_range = 0.5
            else:
                const_range = 0.1

            # make new bounds
            fitter, bounds, grid = self.posterior_setup(li_factor=20, blend_factor=20, const_range=const_range, fit_rv=fit_rv)

            # run mcmc
            self.run_post(spectra, fitter, bounds, grid, fit_rv=fit_rv)

    def posterior(self, spectra):
        '''run ultranest to get posteriors

        Parameters
        ----------
        spectra : dict
            The GALAH dictionary containing the spectra.
        '''

        # set up variables
        self.get_err()
        self.runs = -1
        self.run_res = {}

        # spectra is bad so we skip mcmc
        if self.bad_spec(spectra):
            self.li_fit = None
            self.run_res[self.runs+1] = {'results':None, 'std_li':np.nan, 'time':np.nan, 'bounds':None, 'posterior_good':False, 'due_to_const':False, 'edge_ind':99}
            return None

        fit_rv = False
        self.run_post_widen(spectra, fit_rv=fit_rv)

        # rerun if detect in poorly constrained
        if self.poorly_constrained:
            sample_stats = Stats(sample=self.run_res[self.runs]['results']['samples'][:,0], bounds=self.run_res[self.runs]['bounds'][0])
            err_low = sample_stats.MLE - sample_stats.err_low
            if sample_stats.MLE >= err_low*2:
                fit_rv = True
                self.run_post_widen(spectra, fit_rv=fit_rv)

        # parse results
        std_li = self.run_res[self.runs]['std_li']
        sample = self.run_res[self.runs]['results']['samples']
        bounds = self.run_res[self.runs]['bounds']
        sample_stats = Stats(sample=sample[:,0], bounds=bounds[0])
        MAP, _, _ = self.get_map(sample)
        err = [sample_stats.err_low, sample_stats.err_upp]
        if self.poorly_constrained:
            if fit_rv:
                li_ew, rv, const = MAP
            else:
                li_ew, const = MAP
                rv = 0
            amps = [0]*len(self.narrow_center[1:])
        elif self.mode == 'Gaussian':
            li_ew, *amps, const = MAP
            rv = self.broad_fit['rv']
        elif self.mode == 'Breidablik':
            li_ew, *amps, const = MAP
            rv = self.broad_fit['rv']

        self.li_fit = {'amps':[sample_stats.MLE, *amps], 'const':const, 'std':std_li, 'rv':rv, 'err':err}

        # minchisq
        fitter = self.get_fitter()
        minchisq = chisq(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], fitter.model, fitter.get_init(self.li_fit))
        self.li_fit['minchisq'] = minchisq

    def get_map(self, chain, bins=100):
        '''Get the MAP from the sampled posterior.

        Parameters
        ----------
        chain : ndarray
            The chain from ultranest
        bins : int, optional
            The number of bins to create the histogram with.

        Returns
        -------
        map, inds, uniform : 1darray, 1darray,  bool
            The MAPs for each dimension, and the index that they occur at. If the Li distribution is uniform or not.
        '''

        uniform = False
        params = []
        inds = []
        for i in range(chain.shape[1]):
            sample = chain[:,i]
            hist, edges = np.histogram(sample, bins=bins)
            centers = np.mean([edges[:-1], edges[1:]], axis=0)
            best = centers[np.argmax(hist)]
            params.append(best)
            inds.append(np.argmax(hist))
            # check if Li is uniform, less than 20% of bins are more than 20% away from mean
            if i == 0:
                mean = np.mean(hist)
                extreme_bins = np.sum((hist < mean*0.8) | (hist > mean*1.2))
                if extreme_bins < bins*0.2: # uniform
                    uniform = True
        return np.array(params), np.array(inds), uniform

    def on_edge(self, argmax, bounds, uniform):
        '''Figure out if the MAP occurs on the edge of the sampled posterior.

        Parameters
        ----------
        argmax : 1darray
            The index where the MAP occurs, for all dimensions.
        bounds : 2darray
            The bounded region that the walkers are allowed to be in.
        uniform : bool
            If the Li posterior is uniform or not.

        Returns
        -------
        edge, cont, ind : bool, bool, int
            True if on the edge, otherwise False. True if it's due to the continuum placement, otherwise False. ind is which index triggered it.
        '''

        # check cont
        if argmax[-1] < 5 or argmax[-1] > 94:
            return True, True, -1

        # check Li uniform
        if uniform:
            return True, False, 0

        # indicies for ew
        if self.poorly_constrained:
            inds = [0]
        elif not self.poorly_constrained and self.mode == 'Gaussian':
            inds = list(range(len(argmax)-1))
        elif not self.poorly_constrained and self.mode == 'Breidablik':
            inds = list(range(len(argmax)-1))
            del inds[1]
        # check all ews
        edges = []
        edge_inds = []
        for ind in inds:
            # lower bound
            if argmax[ind] < 5:
                # check that the Breidablik A(Li) bound isn't given by the min ew due to grid
                if ind == 0 and self.mode == 'Breidablik' and bounds[ind][0] <= -self.max_ew:
                    edges.append(False)
                    edge_inds.append(0)
                # lower bound is ok if 0 for blends
                elif ind != 0 and bounds[ind][0] == 0:
                    edges.append(False)
                    edge_inds.append(99)
                else:
                    edges.append(True)
                    edge_inds.append(ind)
            # upper bound
            elif argmax[ind] > 94:
                # check that the Breidablik A(Li) bound isn't given by the max ew due to grid
                if ind == 0 and self.mode == 'Breidablik' and self.max_ew <= bounds[ind][1]:
                    edges.append(False)
                    edge_inds.append(0)
                else:
                    edges.append(True)
                    edge_inds.append(ind)
            # good parameter
            else:
                edges.append(False)
                edge_inds.append(99)
        return any(edges), False, min(edge_inds)

    def get_fitter(self):
        '''Set up the fitter because it's different depending on the mode we are running in.'''

        fit = self.li_fit
        if self.mode == 'Breidablik':
            if not self.poorly_constrained:
                fitter = FitBFixed(center=self.narrow_center[1:], std=self.std, rv=fit['rv'], teff=self.teff, logg=self.logg, feh=self.feh, ew_to_abund=self.ew_to_abund, min_ew=self.min_ew, std_li=fit['std'])
            elif self.poorly_constrained:
                fitter = FitB(self.teff, self.logg, self.feh, self.std, self.ew_to_abund, self.min_ew, std_li=fit['std'])
        elif self.mode == 'Gaussian':
            if not self.poorly_constrained:
                fitter = FitGFixed(center=self.narrow_center, std=self.std, rv=fit['rv'])
            elif self.poorly_constrained:
                fitter = FitG(self.std)
        return fitter

    def plot_broad(self, spectra, show=True, path=None, ax=None):
        '''Plot the broad region and the fits. Meant to be a convenience function for quickly checking the fits are working

        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm)
        show : bool, optional
            Toggle showing the plot, default True.
        path : str, optional
            Path to save fig, if None then it won't save.
        ax : matplotlib.axes, optional
            The axis to plot on, if None, then it will create one to plot on
        '''

        if ax is None:
            axes = plt
        else:
            axes = ax

        # observed spec
        axes.errorbar(spectra['wave_norm'], spectra['sob_norm'], yerr=spectra['uob_norm'], color='black', alpha=0.5, label='observed')
        # fit if well constrained (no fitwell constrained)
        if self.broad_fit is not None:
            if ax is None:
                plt.title(f'{self.sid} {self.std:.4f} {self.snr:.2f}')
            fitter = FitBroad(center=self.broad_center, std=self.std)
            fitter.model(spectra['wave_norm'], [*self.broad_fit['amps'], self.std], plot=True, ax=axes)

        if ax is None:
            plt.xlim(6695, 6719)
            plt.xlabel(r'wavelengths ($\AA$)')
            plt.ylabel('normalised flux')
        axes.legend()
        if path is not None:
            plt.savefig(path, bbox_inches='tight')
        if show:
            plt.show()

    def plot_li(self, spectra, mode='posterior', show=True, path=None, ax=None):
        '''Plot the Li region and the fits..

        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm)
        mode : str, optional
            Plot posterior fits or scipy.minimize fits. Values: posterior, minimize.
        show : bool, optional
            Toggle showing the plot, default True.
        path : str, optional
            Path to save fig, if None then it won't save.
        ax : matplotlib.axes, optional
            The axis to plot on, if None, then it will create one to plot on
        '''

        if ax is None:
            axes = plt
        else:
            axes = ax

        if mode == 'posterior':
            fit = self.li_fit
        elif mode == 'minimize':
            fit = self.li_init_fit

        # observation
        axes.errorbar(spectra['wave_norm'], spectra['sob_norm'] * fit['const'], yerr=spectra['uob_norm'], label='observed', color='black', alpha=0.5)
        if ax is None:
            plt.title(f'{fit["amps"][0]:.4f} {fit["std"]:.1f}')

        # Breidablik
        if self.mode == 'Breidablik':
            if not self.poorly_constrained:
                fitter = FitBFixed(center=self.narrow_center[1:], std=self.std, rv=fit['rv'], teff=self.teff, logg=self.logg, feh=self.feh, ew_to_abund=self.ew_to_abund, min_ew=self.min_ew, std_li=fit['std'])
            elif self.poorly_constrained:
                fitter = FitB(self.teff, self.logg, self.feh, self.std, self.ew_to_abund, self.min_ew, std_li=fit['std'])
            # error region
            if mode == 'posterior':
                if not np.isnan(fit['err'][0]):
                    lower = bline(spectra['wave_norm'], fit['err'][0], fit['std'], fit['rv'], teff=self.teff, logg=self.logg, feh=self.feh, ew_to_abund=self.ew_to_abund, min_ew=self.min_ew)
                else:
                    lower = np.nan
                if not np.isnan(fit['err'][1]):
                    upper = bline(spectra['wave_norm'], fit['err'][1], fit['std'], fit['rv'], teff=self.teff, logg=self.logg, feh=self.feh, ew_to_abund=self.ew_to_abund, min_ew=self.min_ew)
                else:
                    upper = np.nan
        # Gaussian
        elif self.mode == 'Gaussian':
            if not self.poorly_constrained:
                fitter = FitGFixed(center=self.narrow_center, std=self.std, rv=fit['rv'])
            elif self.poorly_constrained:
                fitter = FitG(self.std)
            # error region
            if mode == 'posterior':
                lower = gline(spectra['wave_norm'], fit['err'][0], fit['std'], fit['rv'], center=self.li_center)
                upper = gline(spectra['wave_norm'], fit['err'][1], fit['std'], fit['rv'], center=self.li_center)
        # Voigt TODO: fix this, it doesn't implement poorly constrained, just to make plotting work for now
        elif self.mode == 'Voigt':
            fitter = FitVFixed(center=self.narrow_center, std=self.std, rv=fit['rv'])

        # plot fit
        fitter.model(spectra['wave_norm'], fitter.get_init(fit), plot=True, plot_all=True, ax=axes)

        # error shaded region
        if mode == 'posterior':
            axes.fill_between(spectra['wave_norm'], lower, y2=upper, alpha=0.5)

        # show chisq region
        axes.axvline(self.narrow_center[1]*(1+fit['rv']/_c)-self.std*2)
        axes.axvline(self.narrow_center[-1]*(1+fit['rv']/_c)+self.std*2)

        axes.legend()
        if ax is None:
            plt.xlabel(r'wavelengths ($\AA$)')
            plt.ylabel('normalised flux')
            plt.xlim(6706, 6709.5)
        else:
            axes.set_xlim(6706, 6709.5)
        if path is not None:
            plt.savefig(path, bbox_inches='tight')
        if show:
            plt.show()

    def plot_corner(self, show=True, path=None):
        '''Corner plot for sampled posterior

        Parameters
        ----------
        show : bool, optional
            Toggle showing the plot, default True.
        path : str, optional
            Path to save fig, if None, then will show fig instead.
        '''

        sample = self.run_res[self.runs]['results']
        paramnames = sample['paramnames']
        data = np.array(sample['weighted_samples']['points'])
        weights = np.array(sample['weighted_samples']['weights'])
        cumsumweights = np.cumsum(weights)

        mask = cumsumweights > 1e-4

        fig = corner.corner(data[mask,:], weights=weights[mask],
                      labels=paramnames, show_titles=True, quiet=True, quantiles=[0.5])

        # you've got to be kidding me, the version of corner on avatar doesn't have overplot_lines
        def _get_fig_axes(fig, K):
            if not fig.axes:
                return fig.subplots(K, K), True
            try:
                return np.array(fig.axes).reshape((K, K)), False
            except ValueError:
                raise ValueError(
                    (
                        "Provided figure has {0} axes, but data has "
                        "dimensions K={1}"
                    ).format(len(fig.axes), K)
                )

        none = [None]*(sample['samples'].shape[1] - 1)
        xs = [self.li_init_fit['amps'][0], *none]
        K = len(xs)
        axes, _ = _get_fig_axes(fig, K)
        axes[0,0].axvline(xs[0], alpha=0.5)

        if path is not None:
            plt.savefig(path, bbox_inches='tight')
        if show:
            plt.show()

        return fig

    def save(self, filepath):
        '''Save the fitted results in a dictionary.

        Parameters
        ----------
        filepath : str
            Filepath to saved results
        '''

        names = ['broad_fit', 'broad_center', # broad
                'poorly_constrained',
                'li_init_fit',
                'li_fit', 'narrow_center',
                'mode',
                'norris',
                'run_res'
                ]
        dic = {}
        for name in names:
            try:
                dic[name] = getattr(self, name)
            except:
                dic[name] = None
        np.save(filepath, dic)

    def load(self, filepath):
        '''Load the fitted results into the class.

        Parameters
        ----------
        filepath : str
            Filepath to saved results
        '''

        dic = np.load(filepath, allow_pickle=True).item()
        for name, value in dic.items():
            setattr(self, name, value)
        try:
            self.runs = len(self.run_res.keys())-1
        except:
            self.runs = -1

