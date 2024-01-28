import glob
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from config import *

def read_meta(directory=info_directory):
    '''Get metadata required to run.'''
    # get the columns that are useful and save it as npy format - done for speed
    # file is smaller, and np.load should be quicker than fits
    with pyfits.open(f'{directory}/{DR}') as hdul:
        data = hdul[1].data
        x = np.rec.array([
            data['sobject_id'],
            #data['flag_repeat'],
            data['snr_px_ccd3'],
            #data['rv_galah'],
            #data['e_rv_galah'],
            data['vsini'],
            data['e_vsini'],
            data['teff'],
            data['e_teff'],
            data['logg'],
            data['e_logg'],
            data['fe_h'],
            data['e_fe_h'],
            data['Li_fe'],
            data['e_Li_fe'],
            data['flag_sp'],
            data['flag_fe_h'],
            ],
            dtype=[
                ('sobject_id', int), 
                #('flag_repeat', int),
                ('snr_c3_iraf', np.float64),
                #('rv_galah', np.float64),
                #('e_rv_galah', np.float64),
                ('vbroad', np.float64),
                ('e_vbroad', np.float64),
                ('teff', np.float64),
                ('e_teff', np.float64),
                ('logg', np.float64),
                ('e_logg', np.float64),
                ('fe_h', np.float64),
                ('e_fe_h', np.float64),
                ('Li_fe', np.float64),
                ('e_Li_fe', np.float64),
                ('flag_sp', int),
                ('flag_fe_h', int),
                ]
            )
    np.save(f'{directory}/GALAH_DR.npy', x)

def read_spectra(sobject_id):
    """
    Read in all available in CCD3 and give back a dictionary
    """
    
    DR3 = np.load(f'{info_directory}/GALAH_DR.npy')
    if not (sobject_id in DR3['sobject_id']):
        return None # some stars aren't published. politics, we ignore these stars anyway
        
    # Check if FITS files already available in working directory
    fits_files = f'{working_directory}/{sobject_id}_allstar_fit_spectrum.fits'

    spectrum = dict()
    fits = pyfits.open(fits_files)
            
    # Extract wavelength grid for the reduced spectrum
    spectrum['wave_norm'] = fits[1].data['wave']
    spectrum['sob_norm']  = fits[1].data['sob']
    spectrum['uob_norm']  = fits[1].data['uob']
    #TODO: CDELT is gone
    
    fits.close()

    return spectrum

def cut(spectrum, lower, upper):
    '''Cut the spectrum
    
    spectrum : dictionary
        Output from read_spectra
    lower : float
        The lower value to cut the spectrum to, in Angstroms
    upper : float
        The upper value to cut the specturm to, in Angstroms
    '''
    wl_mask = (lower <= spectrum['wave_norm']) & (spectrum['wave_norm'] <= upper)
    spectrum['wave_norm'] = spectrum['wave_norm'][wl_mask]
    spectrum['sob_norm'] = spectrum['sob_norm'][wl_mask]
    spectrum['uob_norm'] = spectrum['uob_norm'][wl_mask]
    return spectrum

if __name__ == '__main__':
    read_meta()
