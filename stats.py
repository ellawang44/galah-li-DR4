import numpy as np
import math
from scipy.interpolate import CubicSpline, interp1d
from getdist import MCSamples
import matplotlib.pyplot as plt

class Stats:
    def __init__(self, sample=None, bounds=None, x=None, y=None, num=1000):
        if sample is not None: # sampled rvs
            self.xmin, self.xmax = np.min(sample), np.max(sample)
            self.x_interp = np.linspace(self.xmin, self.xmax, num)
            self.calc_cdf(sample)
            self.calc_pdf(sample)
        else: # analytical rvs
            self.x, self.y = x, y
            self.xmin = x[0]
            self.xmax = x[-1]
            self.x_interp = np.linspace(self.xmin, self.xmax, num)
            self.cdf_func = CubicSpline(self.x, np.cumsum(y)/np.sum(y))
            self.pdf_func = CubicSpline(self.x, self.y)
        self.y_interp = self.pdf_func(self.x_interp)
        self.calc_MLE()
        self.calc_err()

    def calc_pdf(self, sample):
        '''Emperical pdf, using boundary corrected KDE.
        
        Parameters
        ----------
        sample : 1darray
            The sample to calculate the pdf for. 
        '''

        mcsamples = MCSamples(samples=sample, names=['Li'], ranges={'Li':[self.xmin, self.xmax]})
        kde = mcsamples.get1DDensityGridData('Li')
        self.pdf_func = lambda x : kde(x)/np.trapz(kde(x), x=x)

    def calc_cdf(self, sample):
        '''Emperical cdf, with linear interpolation. 

        Parameters
        ----------
        sample : 1darray
            The sample to calculate the cdf for. 
        '''
        
        self.cdf_func = interp1d(np.sort(sample), np.linspace(0, 1, len(sample)), fill_value='extrapolate')

    def calc_MLE(self):
        '''Maximum likelihood. 
        '''

        self.MLE_ind = np.argmax(self.y_interp)
        self.MLE = self.x_interp[self.MLE_ind]
    
    def cross(self, x, y, line):
        '''Find where line intersects the numerical function x, y.

        Parameters
        ----------
        x : 1darray
            The x array to the numerical function.
        y : 1darray
            The y array to the numerical function.
        line : float
            The horizontal line that intersects the numerical function.

        Returns
        -------
        xs : 1darray
            The x values where the line intersects the function, linear interpolation is used.  
        '''

        # find places where y crosses the line
        xs = []
        for ind, (left, right) in enumerate(zip(y[:-1], y[1:])):
            if np.min([left, right]) <= line <= np.max([left, right]):
                grad = (right-left)/(x[ind+1]-x[ind])
                inter = left - x[ind]*grad
                x_target = (line-inter)/grad
                xs.append(x_target)
        xs = np.array(xs)
        return xs

    def calc_err(self, threshold=0.005, num=1000):
        '''Calculate the errors. Uses KDE, unless single sided error.

        Parameters
        ----------
        threshold : float, optional
            The tolerance till convergence. Default 0.5% either side of 68%.  
        '''

        def calc_areas(line):
            '''Calculate all of the areas (pairwise intersects).
            '''

            xs = self.cross(self.x_interp, self.y_interp, line)
            lower_xs = xs[xs < self.MLE]
            upper_xs = xs[self.MLE < xs]
            areas = []
            for l in lower_xs:
                for u in upper_xs:
                    area = self.cdf_func(u) - self.cdf_func(l)
                    areas.append([area, l, u])
            return areas
        
        def closest_area(areas):
            '''Give back the area closest to 68%.
            '''

            diffs = [np.abs(area[0] - 0.68) for area in areas]
            return areas[np.argmin(diffs)]
        
        # starting guess
        line = np.max(self.y_interp)*(1-1/num)
        areas = calc_areas(line)
        if len(areas) == 0:
            half = True
            current_area = [0.68]
        else:
            best_area = closest_area(areas)
            current_area = best_area
            half = False

        # iterate till optimal line palcement
        while np.abs(current_area[0] - 0.68) > threshold:
            # find all areas
            areas = calc_areas(line)
            # one of the sides has fallen off
            if len(areas) == 0: 
                half = True
                break
            # find the closest area in this batch
            current_area = closest_area(areas)
            # update best area if this is area is better
            if abs(current_area[0] - 0.68) < abs(best_area[0] - 0.68):
                best_area = current_area
            # walked past optimal
            if all([a[0] > 0.68+threshold for a in areas]): 
                break
            line -= np.max(self.y_interp)/num 
        self.line = line

        # calculate x values from line
        if not half:
            self.err_low = best_area[1]
            self.err_upp = best_area[2]
            self.area = best_area[0]
        # reporting 34% error
        if half:
            if (self.y_interp[0] - self.y_interp[-1]) > 0.5*np.max(self.y_interp):
                self.calc_half(side='upper')
            elif (self.y_interp[-1] - self.y_interp[0]) > 0.5*np.max(self.y_interp):
                self.calc_half(side='lower')
            else:
                self.err_low = np.nan
                self.err_upp = np.nan
                self.area = np.nan

    def calc_half(self, side):
        '''Calculate the lower or upper error as 34% away from the MLE. Set other side to np.nan.

        Parameters
        ----------
        side : str
            The side to calculate the error on. Can be lower or upper.
        '''
        
        # interp1d gives nan if there is more than 1 minimum value, at the minimum value. 
        if self.x_interp[self.MLE_ind] <= self.xmin:
            MLE_cdf = 0
        else:
            MLE_cdf = self.cdf_func(self.x_interp)[self.MLE_ind]
        
        if side == 'lower':
            self.err_upp = np.nan
            if MLE_cdf <= 0.34: # bad distribution
                self.err_low = np.nan
                self.area = np.nan
            else:
                self.err_low = self.cross(self.x_interp, self.cdf_func(self.x_interp), MLE_cdf - 0.34)[0]
                self.area = self.cdf_func(self.MLE) - self.cdf_func(self.err_low)
        elif side == 'upper':
            self.err_low = np.nan
            if MLE_cdf >= (1-0.34): # bad distribution
                self.err_upp = np.nan
                self.area = np.nan
            else:
                self.err_upp = self.cross(self.x_interp, self.cdf_func(self.x_interp), MLE_cdf + 0.34)[0]
                self.area = self.cdf_func(self.err_upp) - self.cdf_func(self.MLE)

