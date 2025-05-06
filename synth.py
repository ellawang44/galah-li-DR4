import numpy as np
from breidablik.interpolate.spectra import Spectra
from breidablik.analysis import read
from astro_tools import SpecAnalysis
from scipy.stats import norm
from astro_tools import vac_to_air
from scipy.interpolate import CubicSpline
from scipy.special import voigt_profile
import matplotlib.pyplot as plt

_c = 299792.458 # speed of light in km s^-1
# optimised from 8s for 100 spectra to 2s - cut mainly, gaussian broadening versions don't make too much of a difference
_spectra = Spectra()
# cut to 6703 - 6712 (a little bit extra for rv shift)
_spectra.cut_models = _spectra.models[136:298]
_wl = vac_to_air(read.get_wavelengths()*10)[136:298]

def bline(x, ew, std, rv, teff, logg, feh, ew_to_abund, min_ew, grid=None):
    '''Li line profiles from breidablik or interpolation grid.

    Parameters
    ----------
    x: 1darray
        The wavelengths to evaluate the spectral line at
    ew : float
        The EW of the line
    std : float
        The standard deviation of the line. If breidablik=True, this is the amount that the std that goes into the Gaussian convolution.
    rv : float
        The radial velocity.
    teff : float, optional
        Used in breidablik, teff of star
    logg : float, optional
        Used in breidablik, logg of star
    feh : float, optional
        Used in breidablik, feh of star
    ew_to_abund : object, optional
        Converting EW to A(Li), used in Breidablik since the input there is A(Li), but the input to this function is EW.
    min_ew : float
        The EW corresponding to A(Li) = -0.5, used for mirroring to emission.
    grid : object, optional
        Grid of cubicsplines to interpolate spectra on, faster than computing from scratch. Default None which doesn't use grid.

    Returns
    -------
    flux : 1darray
        The flux from Breidablik (3D NLTE Li profile).
    '''

    if (grid is None) or (grid.grid is None):
        flux = bprof(ew, std, teff, logg, feh, ew_to_abund, min_ew)
    elif grid.dim == 1:
        assert np.abs(std - grid.std) < 1e-5
        flux = grid.interpolate(ew)
    wl = _wl*(1+rv/_c)
    return CubicSpline(wl, flux)(x)

def bprof(ew, std, teff, logg, feh, ew_to_abund, min_ew):
    '''Li line profiles from breidablik

    Parameters
    ----------
    ew : float
        The EW of the line
    std : float
        The standard deviation of the line. If breidablik=True, this is the amount that the std that goes into the Gaussian convolution.
    teff : float, optional
        Used in breidablik, teff of star
    logg : float, optional
        Used in breidablik, logg of star
    feh : float, optional
        Used in breidablik, feh of star
    ew_to_abund : object
        Converting EW to A(Li), used in Breidablik since the input there is A(Li), but the input to this function is EW.
    min_ew : float
        The EW corresponding to A(Li) = -0.5, used for mirroring to emission.

    Returns
    -------
    flux : 1darray
        The flux from Breidablik (3D NLTE Li profile).
    '''

    if ew >= min_ew:
        ali = ew_to_abund(ew)
        flux = _spectra._predict_flux(teff, logg, feh, [ali])[0]
    elif ew <= -min_ew:
        ali = ew_to_abund(np.abs(ew))
        flux = 2-_spectra._predict_flux(teff, logg, feh, [ali])[0]
    else:
        grid_ews = np.array([-min_ew, min_ew])
        flux = _spectra._predict_flux(teff, logg, feh, -0.5)[0]
        fluxes = np.array([2-flux, flux])
        grads = (fluxes[1] - fluxes[0])/(grid_ews[1] - grid_ews[0])
        intercepts = fluxes[1] - grads*grid_ews[1]
        flux = ew*grads+intercepts
    # gaussian broaden
    if std <= 0:
        flux = flux
    else:
        spec = SpecAnalysis(_wl, flux)
        _, flux = spec._gaussian_broaden(center=6707.814, mode='std', broad=std*_c/6707.814)
    return flux

def vline(x, amp, sigma, gamma, rv, center):
    '''Create a voigt spectral line.

    Parameters
    ----------
    x : 1darray
        The wavelengths to evaluate the spectral line at
    amp : float
        The amplitude to scale the spectral line
    sigma : float
        The standard deviation of the normal distribution
    gamma : float
        The HWHM of the Cauchy distribution
    rv : float
        The radial velocity.
    center : float
        The center that the line is at
    '''

    y = 1-amp*voigt_profile(x-center*(1-rv/_c), sigma, gamma)
    return y

def gline(x, ew, std, rv, center):
    '''Create a Gaussian spectral line.

    Parameters
    ----------
    x : 1darray
        The wavelengths to evaluate the spectral line at
    ew : float
        The EW of the line
    std : float
        The standard deviation of the line.
    rv : float
        The radial velocity.
    center : float
        The center that the line is at.

    Returns
    -------
    y : 1darray
        The spectral line, flux, at the input x wavelengths.
    '''

    y = 1-ew*norm.pdf(x, center*(1+rv/_c), std)
    return y

class Grid:
    '''Interpolation grid for Li over ews.
    '''

    def __init__(self, ewrange, cutoff=4000, **kwargs):
        '''
        Parameters
        ----------
        ewrange : [float, float]
            min ew, max ew
        cutoff : int, optional
            Any value higher will not have a grid computed for it, takes too long.
        **kwargs
            kwargs that go into the breidablik profile function.
        '''

        self.dim = 1
        self.ewnum = max(int(np.ceil((ewrange[1]-ewrange[0])/1e-1)), 3)
        self.ews = np.linspace(ewrange[0], ewrange[1], self.ewnum)
        self.std = kwargs['std']
        if self.ewnum > cutoff:
            self.grid = None
        else:
            self.grid = self.make_grid(**kwargs)

    def make_grid(self, **kwargs):
        '''Make grid of splines for interpolation.

        Parameters
        ----------
        **kwargs
            kwargs that go into the breidablik profile function.

        Returns
        -------
        grid : list of list of splines
            Grid of splines for interpolation.
        '''

        fluxes = [bprof(ew, **kwargs) for ew in self.ews]
        grid = [CubicSpline(self.ews, f) for f in np.array(fluxes).T]

        return grid

    def interpolate(self, ew):
        '''Interpolate to profile on grid.

        Parameters
        ----------
        ew : float
            EW of the line

        Returns
        -------
        int_flux : 1darray
            The interpolated flux.
        '''

        if self.grid is None:
            return None

        flux = [cs(ew) for cs in self.grid]

        return np.array(flux)


# convert between std and fwhm
std_to_fwhm = lambda x: 2*np.sqrt(2*np.log(2))*x
fwhm_to_std = lambda x: x/(2*np.sqrt(2*np.log(2)))

def encompass(y, y_target, offset=0):
    '''Find the index of the elements which encompass the target

    Parameters
    ----------
    y : 1darray
        The y array of values, needs to be monotonic.
    y_target : float
        The target y value that you want values surrounding.
    offset : int
        The offset for the indicies, if you only give half the y array for monotonic reasons.

    Returns
    -------
    left_ind, right_ind : int, int
        The left and right index of the ys that encompass the y_target value.
    '''

    x = np.arange(0, len(y))
    less = x[y < y_target]
    more = x[y > y_target]
    if less[0] < more[0]:
        left_ind = less[-1]
        right_ind = more[0]
    elif more[0] < less[0]:
        left_ind = more[-1]
        right_ind = less[0]
    return left_ind + offset, right_ind + offset

def measure_fwhm(x, y, ratio=0.5, threshold=0.001):
    '''Measure fwhm from input line.

    Parameters
    ----------
    x : 1darray
        x values of the line
    y : 1darray
        y values of the line
    ratio : float
        The ratio of the depth to find the fwhm at.
    threshold : float
        The threshold to achieve before linear interpolating. Smaller values are more accurate but take longer to compute.

    Returns
    -------
    fwhm : float
        The full width half max measured.
    '''

    # setup
    cs = CubicSpline(x, y)
    ind_center = np.argmin(y)
    ymax = np.min(y)
    yhalf = 1-(1-ymax)*ratio
    # find inds
    left_ind1, left_ind2 = encompass(y[:ind_center], yhalf, offset=0)
    right_ind1, right_ind2 = encompass(y[ind_center:], yhalf, offset=ind_center)
    # window search
    left_x, h = window_search([x[left_ind1], x[left_ind2]], cs, yhalf, threshold=threshold, bound_y=[y[left_ind1], y[left_ind2]])
    right_x, h2 = window_search([x[right_ind1], x[right_ind2]], cs, yhalf, threshold=threshold, bound_y=[y[right_ind1], y[right_ind2]])
    return right_x - left_x

def calc_std(x, ew, fwhm, teff, logg, feh, ew_to_abund, min_ew, ratio=0.5, ex_step=0.1):
    '''Calculate the standard deviation required to achieve the fwhm for a Li line.

    Parameters
    ----------
    x : 1darray
        Wavelengths
    ew : float
        The EW of the Li line
    fwhm : float
        The fwhm of the Li line
    teff : float, optional
        Used in breidablik, teff of star
    logg : float, optional
        Used in breidablik, logg of star
    feh : float, optional
        Used in breidablik, feh of star
    ew_to_abund : object, optional
        Converting EW to A(Li), used in Breidablik since the input there is A(Li), but the input to this function is EW.
    min_ew : float
        The EW corresponding to A(Li) = -0.5, used for mirroring to emission.
    ratio : float, optional
        The ratio of the depth to find the fwhm at.
    ex_step : float, optional
        The step to extend the search by.

    Returns
    -------
    new_std : float
        The std required to achieve the fwhm
    '''

    func = lambda std: measure_fwhm(x, bprof(ew, std, teff, logg, feh, ew_to_abund, min_ew), threshold=0.001, ratio=ratio)
    # estimate std
    base_fwhm = func(0)
    std = fwhm_to_std(np.sqrt(fwhm**2 + base_fwhm**2))
    # find 2 stds encompassing the fwhm
    new_fwhm = func(std)
    if fwhm < new_fwhm:
        low_std = std-ex_step
        low_fwhm = func(low_std)
        while fwhm < low_fwhm:
            low_std -= ex_step
            low_fwhm = func(low_std)
            # Li line thermal width is already broader than the fwhm
            if low_std <= 0:
                return 0
        upp_std = std
        upp_fwhm = new_fwhm
    elif new_fwhm < fwhm:
        low_std = std
        low_fwhm = new_fwhm
        upp_std = std+ex_step
        upp_fwhm = func(upp_std)
        while upp_fwhm < fwhm:
            upp_std += ex_step
            upp_fwhm = func(upp_std)
    # window search
    new_std, _ = window_search([low_std, upp_std], func, fwhm, threshold=0.00001, bound_y=[low_fwhm, upp_fwhm])
    return new_std

def window_search(bound_x, func, target_y, threshold=0.001, bound_y=None):
    '''The window search algorithm, with last step as a linear interpolation. Will probably fail if function is not monotonic within the bounds.

    Parameters
    ----------
    bound_x : (float, float)
        The x values that bound the target_y value
    func : function
        The function that takes in x values and returns y values.
    target_x : float
        The y value that you want to reach.
    threshold : float, optional
        The tolerance threshold for where the function is locally linear. Default 0.001
    bound_y : (float, float), optional
        The y values corresponding to bound_x, the function will be called with bound_x if not given.

    Returns
    -------
    new_x, new_y : (float, float)
        The x and y value that are within threshold to the target y value.
    '''

    # set up
    left_x, right_x = bound_x
    if bound_y is None:
        left_y, right_y = func(left_x), func(right_x)
    else:
        left_y, right_y = bound_y

    # double check that target is bounded
    assert ((left_y <= target_y) & (target_y <= right_y)) | ((right_y <= target_y) & (target_y <= left_y))
    new_x = (left_x + right_x)/2
    new_y = func(new_x)
    # iterate
    while np.abs(new_y - target_y) > threshold:
        if target_y < new_y:
            if left_y < right_y:
                right_x = new_x
            else:
                left_x = new_x
                left_y = new_y
        else:
            if left_y < right_y:
                left_x = new_x
                left_y = new_y
            else:
                right_x = new_x
                right_y = new_y
        new_x = (left_x + right_x)/2
        new_y = func(new_x)
    # linear interpolate
    grad = (right_x - left_x)/(right_y - left_y)
    inter = left_x - grad*left_y
    target_x = grad*target_y+inter
    return target_x, func(target_x)

