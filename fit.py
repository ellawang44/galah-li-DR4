import copy
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from synth import bline, gline, _c, _wl, std_to_fwhm, calc_std


def chisq(wl_obs, flux_obs, flux_err, model, params, bounds=None, wl_left=None, wl_right=None):
    '''Calculate the chisq with bounds. If value is out of bounds, then chisq is inf. Note parameter and bounds need to have same ordering. 
    wl_left and wl_right need to be both given, or else ignored, it is the region in which to compute the chisq value. Needed for continuum normalisation constant, set to the extreme narrow region centers +- std, including poorly constrained stars for consistency. 
    
    Parameters
    ----------
    wl_obs : 1darray
        observed wavelengths
    flux_obs : 1darray
        observed flux
    flux_err : 1darray
        observed flux error
    model : object
        Model to evaluate 
    params : 1darray
        Parameters to the model
    bounds : 1darray, optional
        Bounds for the parameters
    wl_left : float, optional
        Left wl bound to compute chisq over. 
    wl_right : float, optional
        Right wl bound to compute chisq over. 

    Returns
    -------
    chisq : float
        The chisq value 
    '''
   
    if bounds is not None:
        assert len(params) == len(bounds)

        for p, (l, r) in zip(params, bounds):
            if (p < l) or (r < p):
                return np.inf
    
    if (wl_left is not None) and (wl_right is not None):
        mask = (wl_left <= wl_obs) & (wl_obs <= wl_right)
        wl_obs = wl_obs[mask]
        flux_obs = flux_obs[mask]
        flux_err = flux_err[mask]

    return np.sum(np.square((model(wl_obs, params) - flux_obs)/flux_err))


class FitG:
    '''Fits rv and a EW for each center given. 
    Only Li line, poorly constrained, when sp is nan.
    '''

    def __init__(self, std, rv_lim=None):
        '''Optional parameters are not needed if only using the model and not fitting. 
        
        Parameters
        ----------
        std : float
            The std to use for the fit. From GALAH.
        rv_lim : float, optional
            The limit on rv, mirrored limit on either side, it is based on galah vsini.
        '''

        self.std = std
        # don't need if using model
        self.rv_lim = rv_lim
        # for mcmc
        self.std_li = std

    def get_init(self, init):
        '''Construct init list from dict'''

        return [init['amps'][0], init['rv'], init['const']]

    def fit(self, wl_obs, flux_obs, flux_err, init):
        '''Fit EW, rv, and continuum of observed spectrum.
        
        Parameters
        ----------
        wl_obs : 1darray
            observed wavelengths
        flux_obs : 1darray
            observed flux
        flux_err : 1darray
            observed flux error
        init : list
            The initial guess, use get_init to format from dictionary 

        Returns
        -------
        fit : dict
            Fitted parameters
        '''
        
        # construct bounds
        bounds = [(-np.inf, np.inf)] # Li EW can be negative
        bounds.append((-self.rv_lim, self.rv_lim)) # given in init
        bounds.append((0.5, 1.5)) # continuum normalisation constant
        
        # fit
        func = lambda x: chisq(wl_obs, flux_obs, flux_err, self.model, x, bounds, wl_left=6706.730*(1+x[1]/_c)-self.std*2, wl_right=6708.961*(1+x[1]/_c)+self.std*2)
        res = minimize(func, self.get_init(init), method='Nelder-Mead')

        return {'li':res.x[0], 'std_li':np.std, 'rv':res.x[1], 'amps':[], 'const':res.x[2], 'minchisq':res.fun}

    def model(self, wl_obs, params, plot=False, ax=None, plot_all=False, grid=None):
        '''One Gaussian for Li.

        wl_obs : np.array
            observed wavelengths
        params : np.array
            Parameters to the model. Use get_init to format from dictionary.
        plot : bool
            If True, turns on plotting.
        ax : matplotlib.axes, optional
            The axis to plot on, if None, then it's the default one. 
        plot_all : bool
            Not used, only here to be consistent with other classes.
        grid : object
            Not used.

        Returns
        -------
        y : 1darray
            The synthetic spectra model evaluated at given parameters. 
        '''
        
        ew, offset, const = params

        y = gline(wl_obs, ew, self.std, offset, center=6707.814)
        
        # plot
        if plot:
            if ax is None:
                ax = plt
            ax.plot(wl_obs, y, label='fit')
            ax.axvline(6707.814*(1+offset/_c), linestyle='--')
        
        y /= const
       
        return y


class FitGFixed:
    '''Fits EW for each center given. rv is fixed from broad region.
    For narrow region, constrained, with nan sp.
    '''

    def __init__(self, center, std, rv):
        '''Optional parameters are not needed if only using the model and not fitting. 
        
        Parameters
        ----------
        center : 1darray
            The line centers to be fitted. This should be np.array([6707.814, 6706.730, 6707.433, 6707.545, 6708.096, 6708.961])
        std : float
            The std to use for the fit. From GALAH.
        rv : float
            The rv found from the broad region.
        '''

        self.center = center
        self.std = std
        self.rv = rv
        # for mcmc
        self.std_li = std

    def get_init(self, init):
        '''Construct init list from dict'''
        return [*init['amps'], init['const']]

    def fit(self, wl_obs, flux_obs, flux_err, init):
        '''Fit EW and continuum of observed spectrum.
        
        Parameters
        ----------
        wl_obs : 1darray
            observed wavelengths
        flux_obs : 1darray
            observed flux
        flux_err : 1darray
            observed flux error
        init : list
            The initial guess, use get_init to format from dictionary 

        Returns
        -------
        fit : dict
            Fitted parameters
        '''

        # construct bounds
        bounds = [(-np.inf, np.inf)] # Li EW can be negative
        bounds.extend([(0, np.inf) for _ in range(len(init['amps'])-1)]) # positive finite EW
        bounds.append((0.5, 1.5)) # continuum normalisation constant
        
        # fit
        func = lambda x: chisq(wl_obs, flux_obs, flux_err, self.model, x, bounds, wl_left=6706.730*(1+self.rv/_c)-self.std*2, wl_right=6708.961*(1+self.rv/_c)+self.std*2)
        res = minimize(func, self.get_init(init), method='Nelder-Mead')

        return {'li':res.x[0], 'std_li':self.std, 'const':res.x[-1], 'amps':res.x[1:-1], 'rv':self.rv, 'minchisq':res.fun}

    def model(self, wl_obs, params, plot=False, ax=None, plot_all=False, grid=None):
        '''Multiplying Gaussians together with a common std. 

        wl_obs : np.array
            observed wavelengths
        params : np.array
            Parameters to the model. Use get_init to format from dictionary.
        plot : bool
            If True, turns on plotting.
        ax : matplotlib.axes, optional
            The axis to plot on, if None, then it's the default one. 
        plot_all : bool
            If True, plot each gaussian. Or else plot only final model.
        grid : object
            Not used. 

        Returns
        -------
        y : 1darray
            The synthetic spectra model evaluated at given parameters. 
        '''

        if plot:
            if ax is None:
                ax = plt
        
        *ews, const = params
        y = np.ones(len(wl_obs))
        
        for a, c in zip(ews, self.center):
            y1 = gline(wl_obs, a, self.std, self.rv, center=c)
            if plot_all:
                ax.plot(wl_obs, y1)
            y *= y1

        if plot:
            ax.plot(wl_obs, y, label='fit')
            ax.axvline(6707.814*(1+self.rv/_c), linestyle='--')
       
        y /= const

        return y


class FitB:
    '''Fits Li EW, rv simultaneously. 
    Only Li line, for poorly constrained stars, Breidablik line profile
    '''

    def __init__(self, teff, logg, feh, std, ew_to_abund, min_ew, max_ew=None, std_li=None, rv_lim=None, ratio=0.75):
        '''Optional parameters are not needed if only using the model and not fitting.
        
        Parameters
        ----------
        teff : float
            Used in breidablik, teff of star
        logg : float
            Used in breidablik, logg of star 
        feh : float
            Used in breidablik, feh of star 
        ew_to_abund : object
            Converting REW to A(Li), used in Breidablik since the input there is A(Li), but the input to this function is EW
        min_ew : float 
            The EW at A(Li) = -0.5 to mirror to emission on
        max_ew : float, optional
            The maximum EW that is allowed
        std_li : float, optional
            The std that Li needs to achieve the input std, will be calculated if not given.
        rv_lim : float, optional
            The limits for rv, based on galah vsini.
        ratio : float, optional
            The ratio of the Li line at which to match the fwhm for.
        '''

        self.teff = teff
        self.logg = logg
        self.feh = feh
        self.std = std
        self.ew_to_abund = ew_to_abund
        self.min_ew = min_ew
        # don't need if using model
        self.max_ew = max_ew
        self.rv_lim = rv_lim
        # calculate parameters for Li fwhm
        self.ratio = ratio
        self.fwhm = std_to_fwhm(std)*0.64423 # this is full width 3/4 max
        self.small_ew = 10**-6*6707.814
        if std_li is None:
            self.std_li = calc_std(_wl, self.small_ew, self.fwhm, self.teff, self.logg, self.feh, self.ew_to_abund, self.min_ew, ratio=self.ratio)
        else:
            self.std_li = std_li

    def get_init(self, init):
        '''Construct init list from dict'''

        return [init['amps'][0], init['rv'], init['const']]

    def fit(self, wl_obs, flux_obs, flux_err, init):
        '''Fit Li EW, Li std of observed spectrum.

        Parameters
        ----------
        wl_obs : np.array
            observed wavelengths
        flux_obs : np.array
            observed flux
        flux_err : np.array
            observed flux error
        init : list
            The initial guess, use get_init to format from dictionary 
        
        Returns
        -------
        fit : dict
            Fitted parameters
        '''
        
        # construct bounds
        bounds = [(-self.max_ew, self.max_ew), # based on cogs
                (-self.rv_lim, self.rv_lim), # based on galah vsini, except in km/s
                (0.5, 1.5)] # continuum normalisation constant

        func = lambda x: chisq(wl_obs, flux_obs, flux_err, self.model, x, bounds, wl_left=6706.730*(1+x[1]/_c)-self.std*2, wl_right=6708.961*(1+x[1]/_c)+self.std*2)
        res = minimize(func, self.get_init(init), method='Nelder-Mead')
       
        return {'li':res.x[0], 'std_li':self.std_li, 'const':res.x[-1], 'amps':[], 'rv':res.x[-2], 'minchisq':res.fun}

    def model(self, wl_obs, params, plot=False, ax=None, plot_all=False, grid=None):
        '''Breidablike ilne profile, with Gaussian broadening.

        wl_obs : np.array
            observed wavelengths
        params : np.array
            Parameters to the model. Use get_init to format from dictionary.
        plot : bool
            If True, turns on plotting.
        ax : matplotlib.axes, optional
            The axis to plot on, if None, then it's the default one. 
        plot_all : bool
            This isn't used, it's just there to be consistent with the other classes.
        grid : object, optional
            The grid to interpolate on, speeds up interpolation. Look at Grid in synth.py
        
        Returns
        -------
        y : 1darray
            The synthetic spectra model evaluated at given parameters. 
        '''
    
        ews, offset, const = params
        y = bline(wl_obs, ews, self.std_li, offset, teff=self.teff, logg=self.logg, feh=self.feh, ew_to_abund=self.ew_to_abund, min_ew=self.min_ew, grid=grid)

        if plot:
            if ax is None:
                ax = plt
            ax.plot(wl_obs, y, label='fit')
            ax.axvline(6707.814*(1+offset/_c), linestyle='--')
        
        y /= const

        return y


class FitBFixed:
    '''Fits Li EW, other ews simltaneously based on the centers given. rv is fixed from broad region.
    For constrained stars, Breidablik line profile
    '''
    
    def __init__(self, center, std, rv, teff, logg, feh, ew_to_abund, min_ew, max_ew=None, std_li=None, ratio=0.75):
        '''Optional parameters are not needed if only using the model and not fitting.
        
        Parameters
        ----------
        center : float
            The center that the blended lines (no Li) is at. The Li line center is already given by Breidablik. Input for this project should be np.array([6706.730, 6707.433, 6707.545, 6708.096, 6708.961])
        std : float
            The std found from galah, used for Gaussians (non-Li lines).
        rv : float
             The rv found from the broad region, used for the whole model.
        teff : float
            Used in breidablik, teff of star
        logg : float
            Used in breidablik, logg of star 
        feh : float
            Used in breidablik, feh of star 
        ew_to_abund : object
            Converting REW to A(Li), used in Breidablik since the input there is A(Li), but the input to this function is EW
        min_ew : float
            The EW at A(Li) = -0.5 to mirror to emission
        max_ew : float, optional
            The maximum EW that is allowed.
        std_li : float, optional
            The std that Li needs to achieve the input std, will be calculated if not given. 
        '''
        
        self.center = center
        self.std = std
        self.rv = rv
        self.teff = teff
        self.logg = logg
        self.feh = feh
        self.std = std
        self.ew_to_abund = ew_to_abund
        # don't need if using model
        self.max_ew = max_ew
        self.min_ew = min_ew
        # calculate parameters for Li fwhm
        self.ratio = ratio
        self.fwhm = std_to_fwhm(std)*0.64423 # this is full width 3/4 max
        self.small_ew = 10**-6*6707.814
        if std_li is None:
            self.std_li = calc_std(_wl, self.small_ew, self.fwhm, self.teff, self.logg, self.feh, self.ew_to_abund, self.min_ew, ratio=self.ratio)
        else:
            self.std_li = std_li

    def get_init(self, init):
        '''Construct init list from dict'''
        return [*init['amps'], init['const']]

    def fit(self, wl_obs, flux_obs, flux_err, init):
        '''Fit ews of observed spectrum.

        Parameters
        ----------
        wl_obs : np.array
            observed wavelengths
        flux_obs : np.array
            observed flux
        flux_err : np.array
            observed flux error
        init : list
            The initial guess, use get_init to format from dictionary 
        
        Returns
        -------
        fit : dict
            Fitted parameters
        '''

        bounds = [(-self.max_ew, self.max_ew)] # based on cog
        bounds.extend([(0, np.inf) for _ in range(len(init['amps'])-1)]) # positive finite EW
        bounds.append((0.5, 1.5)) # continuum normalisation constant

        func = lambda x: chisq(wl_obs, flux_obs, flux_err, self.model, x, bounds, wl_left=6706.730*(1+self.rv/_c)-self.std*2, wl_right=6708.961*(1+self.rv/_c)+self.std*2)
        res = minimize(func, self.get_init(init), method='Nelder-Mead')
        
        return {'li':res.x[0], 'std_li':self.std_li, 'const':res.x[-1], 'amps':res.x[1:-1], 'rv':self.rv, 'minchisq':res.fun}

    def model(self, wl_obs, params, plot=False, ax=None, plot_all=False, grid=None):
        '''Gaussians multiplied together with Breidablik line profile.

        wl_obs : np.array
            observed wavelengths
        params : np.array
            Parameters to the model. Use get_init to format from dictionary.
        plot : bool
            If True, turns on plotting.
        ax : matplotlib.axes, optional
            The axis to plot on, if None, then it's the default one. 
        plot_all : bool
            If True, plot each gaussian. Or else plot only final model.
        grid : object, optional
            The grid to interpolate on, speeds up interpolation. Look at Grid1D and Grid2D in synth.py
        
        Returns
        -------
        y : 1darray
            The synthetic spectra model evaluated at given parameters. 
        '''

        if plot:
            if ax is None:
                ax = plt

        ali, *ews, const = params
        y = bline(wl_obs, ali, self.std_li, self.rv, teff=self.teff, logg=self.logg, feh=self.feh, ew_to_abund=self.ew_to_abund, min_ew=self.min_ew, grid=grid) 
        if plot:
            ax.plot(wl_obs, y, label='Li')
        
        for a, c in zip(ews, self.center):
            y1 = gline(wl_obs, a, self.std, self.rv, center=c)
            if plot_all:
                ax.plot(wl_obs, y1)
            y *= y1
        
        # plot
        if plot:
            ax.plot(wl_obs, y, label='fit')
            ax.axvline(6707.814*(1+self.rv/_c), linestyle='--')
        
        y /= const
        
        return y


class FitBroad:
    '''Fits rv and a EW for each center given. 
    For broad region to measure rv.
    '''

    def __init__(self, center, std, rv_lim=None):
        '''Optional parameters are not needed if only using the model and not fitting. 
        
        Parameters
        ----------
        center : 1darray
            The line centers to be fitted. This should be np.array([6696.085, 6698.673, 6703.565, 6705.101, 6710.317, 6711.819, 6713.095, 6713.742, 6717.681])
        std : float
            The sqrt(vbroad^2+PSF^2) from galah.
        rv_lim : float, optional
            The limit on rv, mirrored limit on either side, it is based on the galah vsini, except in km/s.
        '''

        self.center = center
        self.std = std
        # don't need if using model
        self.rv_lim = rv_lim

    def get_init(self, init):
        '''Construct init list from dict'''
        return [*init['amps'], init['rv']]

    def fit(self, wl_obs, flux_obs, flux_err, init):
        '''Fit rv of observed spectrum.
        
        Parameters
        ----------
        wl_obs : 1darray
            observed wavelengths
        flux_obs : 1darray
            observed flux
        flux_err : 1darray
            observed flux error
        init : list
            The initial guess, use get_init to format from dictionary 

        Returns
        -------
        fit : dict
            Fitted parameters
        '''
        
        # poorly constrained star
        if init is None:
            return None

        # construct bounds
        bounds = [(0, np.inf) for _ in range(len(init['amps']))] # positive finite EW
        bounds.append((-self.rv_lim, self.rv_lim)) # given in init
        
        # fit
        func = lambda x: chisq(wl_obs, flux_obs, flux_err, self.model, x, bounds)
        res = minimize(func, self.get_init(init), method='Nelder-Mead')

        return {'amps':res.x[:-1], 'std':self.std, 'rv':res.x[-1], 'minchisq':res.fun, 'bounds':bounds}

    def model(self, wl_obs, params, plot=False, ax=None, plot_all=False):
        '''Multiplying Gaussians together with a common std and rv. 

        wl_obs : np.array
            observed wavelengths
        params : np.array
            Parameters to the model. Use get_init to format from dictionary.
        plot : bool
            If True, turns on plotting.
        ax : matplotlib.axes, optional
            The axis to plot on, if None, then it's the default one. 
        plot_all : bool
            If True, plot each gaussian. Or else plot only final model.

        Returns
        -------
        y : 1darray
            The synthetic spectra model evaluated at given parameters. 
        '''
        
        if plot:
            if ax is None:
                ax = plt
        
        *ews, offset = params
        y = np.ones(len(wl_obs))
        
        for a, c in zip(ews, self.center):
            y1 = gline(wl_obs, a, self.std, offset, center=c)
            if plot_all:
                ax.plot(wl_obs, y1)
            y *= y1
        
        # plot
        if plot:
            ax.plot(wl_obs, y, label='fit')
        
        return y


def pred_amp(wl_obs, flux_obs, flux_err, centers, rv=0, perc=95, set_cont=False):
    '''Get the amplitudes for the initial guess. 

    Parameters
    ----------
    wl_obs : np.array
        observed wavelengths
    flux_obs : np.array
        observed flux
    flux_err : np.array
        observed flux error
    centers : 1darray
        The centers the lines are at -- these are the wls used to find the amplitudes
    rv : float, optional
        rv shift, used to shift the centers. Default 0. 
    perc : float, optional
        The percentile to use for the continuum estimation. Default 95
    set_cont : bool, optional
        Set continuum 1. Default False, which means continuum is estimated.

    Returns
    -------
    amps, err, cont : 1darray, 1darray, float
        Amplitudes of observed spectra at centers, set to 0 if negative; the flux errors at those amplitudes; the continuum placement.
    '''

    # pred continuum
    if set_cont:
        cont = 1
    else:
        cont = np.percentile(flux_obs, perc)
    # predict amplitudes
    inds = np.array([np.argmin(np.abs(wl_obs - i*(1+rv/_c))) for i in centers])
    amps = (1 - (flux_obs/cont)[inds])*1.01 # bit bigger because sampling
    amps[amps < 0] = 0 # set negative amp to 0, chisq is inf otherwise
    err = flux_err[inds]
    return amps, err, 1/cont

def check_mp(amps, err):
    '''check if poorly constrained star. Criteria is <3 amplitudes above error.

    Parameters
    ----------
    amps : 1darray
        amplitudes 
    err : 1darray
        errors at the amplitudes

    Returns
    -------
    mp : bool
        If True, then this is a poorly constrained star (less than 3 lines detected)
    '''

    mask = amps > err
    if np.sum(mask) < 3: # 3 lines detection is arbitrary
        return True
    else:
        return False

def cross_correlate(wl, flux, centers, amps, std, rv):
    '''Calculate the cross correlation between template and obs flux.
    
    Parameters
    ----------
    wl : 1darray
        observed wavelengths
    flux : 1darray
        observed flux
    centers : float
        centers the lines are at
    amps : 1darray
        EWs of model (multiplying Gaussians together). 
    std : float
        width
    rv : float
        radial velocity shift

    Returns
    -------
    cc : float
        The cross correlation between the observed spectrum and the model spectrum
    '''
    
    fit_all = FitBroad(center=centers, std=std)
    template = fit_all.model(wl, [*amps, std, rv])
    cc = np.sum(template*flux)
    return cc

def cc_rv(wl, flux, centers, amps, std, rv_init, rv_lim):
    '''Get best rv from cross correlation. Searches 10 km/s to either side of rv_init.

    Parameters
    ----------
    wl : 1darray
        observed wavelengths
    flux : 1darray
        observed flux
    centers : 1darray
        centers of the lines
    amps : 1darray
        EWs of model (multiplying Gaussians together).
    std : float
        width
    rv_init : float
        rv to search around
    rv_lim : float
        limit to the rvs searched through

    Returns
    -------
    rv : float
        rv from cross correlation (2dp accuracy)
    '''

    # rv 10 km/s is shifting about the line width.
    rvs = np.linspace(rv_init-10, rv_init+10, 2000) # accurate to 2nd dp
    rvs = rvs[np.abs(rvs)<rv_lim]# filter out values beyond rv_lim
    ccs = [cross_correlate(wl, flux, centers, amps, std, rv) for rv in rvs]
    return rvs[np.argmax(ccs)]

def filter_spec(spec, sigma=5):
    '''filter weird parts of the spectrum out. Sets extremely small flux errors to the median flux error. GALAH DR3 can underestimate flux error by more than an order of magnitude, massively biasing fits. Did not check if this problem persists in DR4. 

    Parameters
    ----------
    spec : dict
        Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm)

    Returns
    -------
    spec : dict
        Filtered spectrum, in same dictionary as input.
    '''

    # filter negative flux
    mask = spec['sob_norm'] >= 0
    # filter sigma too small, if too small change to medium
    medium_sig = np.nanmedian(spec['uob_norm'])
    mask_medium = spec['uob_norm'] < medium_sig/10 # allow 1 order of magnitude
    spec['uob_norm'][mask_medium] = medium_sig
    # filter flux which sigma*error above 1
    #mask = mask & (spec['sob_norm'] < (1 + spec['uob_norm']*sigma))
    # this filter results in things being overfiltered
    # write
    spec['uob_norm'] = spec['uob_norm'][mask]
    spec['sob_norm'] = spec['sob_norm'][mask]
    spec['wave_norm'] = spec['wave_norm'][mask]
    return spec

def broken_spec(spec):
    '''Filter broken spectra out. 
    '''

    # weird normalisation, M-type stars suck.
    if np.all(spec['sob_norm'] < 0.5):
        return True
    # not enough pixels
    if len(spec['wave_norm']) < 50:
        return True
    return False

def amp_to_init(amps, std, const, rv=0):
    '''convert amplitudes to initial guess (ew & include std, rv)
    
    Parameters
    ----------
    amps : 1darray
        amplitudes
    std : float
        std of the gaussians
    const : float
        constant for continuum normalisation
    rv : float, optional
        radial velocity of the star

    Returns
    -------
    init : dict
        The initial guess. ews are from amps and std. 
    '''
    
    init = list(np.array(amps)*np.sqrt(2*np.pi)*std) # amp to ew
    return {'amps':init, 'std':std, 'rv':rv, 'const':const}

def iter_fit(wl, flux, flux_err, center, std, rv_lim):
    '''Iteratively fit the broad region. The fit is only as good as the rv, which is only as good as the ews. Hence iteratively fit between cross correlated rv and ews. Gives up after 5 iterations and returns the initial fit.

    Parameters
    ----------
    wl : 1darray
        observed wavelengths
    flux : 1darray
        observed flux
    flux_err : 1darray
        observed flux error
    center : 1darray
        centers of the lines to be fitted. Should be np.array([6696.085, 6698.673, 6703.565, 6705.101, 6710.317, 6711.819, 6713.095, 6713.742, 6717.681])).
    std : float
        The std from GALAH in \AA
    rv_lim : float
        The limit on rv, mirrored limit on either side, it is based on galah vbroad values..
    
    Returns
    -------
    res : dict
        Iteratively fitted ews, rv
    '''

    # get initial rv
    fitter = FitBroad(center=center, std=std, rv_lim=rv_lim)
    amps, _, _ = pred_amp(wl, flux, flux_err, center, set_cont=True)
    res = amp_to_init(amps, std, 1, rv=0)
    init_rv = cc_rv(wl, flux, center, res['amps'], res['std'], res['rv'], rv_lim)
    
    # get initial amp
    amps, err, _ = pred_amp(wl, flux, flux_err, center, rv=init_rv, set_cont=True)
    init = amp_to_init(amps, std, 1, rv=init_rv)

    # check poorly constrained star
    if check_mp(amps, err):
        return None

    # get good initial fit
    res = fitter.fit(wl, flux, flux_err, init)
    if res is None: # poorly constrained star
        return None
    initial_res = copy.copy(res)

    # iterate
    iterations = 1
    while np.abs(init_rv - res['rv']) > 0.1:
        init_rv = cc_rv(wl, flux, center, res['amps'], res['std'], res['rv'], rv_lim)
        res = fitter.fit(wl, flux, flux_err, init={'amps':res['amps'], 'rv':init_rv})
        iterations += 1
        if iterations >= 5:
            res = initial_res
            print('iterations over 5')
            break
    
    # check poorly constrained again because some fits are bad
    if check_mp(res['amps'], err):
        return None
    
    return res

