# if a file doesn't exist, run setup.sh and config.py

from read import read_spectra, read_meta, cut
import os
import numpy as np
import matplotlib.pyplot as plt
from fit import filter_spec, broken_spec
from synth import fwhm_to_std
import argparse
from run import FitSpec
from config import *
import copy
from multiprocessing import Pool

# get metadata values
data = np.load(f'{info_directory}/GALAH_DR.npy')
sobject_id = data['sobject_id']
SNR = data['snr_c3_iraf']
teff = data['teff']
logg = data['logg']
feh = data['fe_h'] 
vbroad = fwhm_to_std(data['vbroad']*6707.814/299792.458) # \AA, std
fwhm = np.load(f'{info_directory}/fwhm.npy')
assert np.allclose(fwhm[:,0], sobject_id) # required so the fwhm values are matched to the correct sobject_ids
galah_psf = fwhm_to_std(fwhm[:,1]) # \AA, std
galah_std = np.sqrt(np.square(vbroad) + np.square(galah_psf)) # \AA, std

def run(i):
    print(i)

    spectra = read_spectra(i)
    if spectra is None:
        return None

    # cut to broad region for std fitting
    spectra = cut(spectra, 6695, 6719)
    spectra = filter_spec(spectra) 

    spectra_broad = copy.deepcopy(spectra)
    if len(spectra['wave_norm']) == 0: # cut doesn't work if this is []
        return None

    # cut to Li region for detailed fitting
    spectra = cut(spectra, 6704, 6711)
    if broken_spec(spectra):
        return None

    # identify object
    ind = np.where(i==sobject_id)[0][0]
    rv_lim = galah_std[ind]/6707.814*299792.458 # km/s

    # no galah fwhm, skip
    if np.isnan(galah_std[ind]):
        return None

    # fitting
    fitspec = FitSpec(std=galah_std[ind], snr=SNR[ind], sid=i, teff=teff[ind], logg=logg[ind], feh=feh[ind], rv_lim=rv_lim)
    # load fit
    if os.path.exists(f'{info_directory}/fits/{i}.npy') and args.load_fit:
        fitspec.load(f'{info_directory}/fits/{i}.npy')
        if args.save_fit:
            fitspec.save(f'{info_directory}/fits/{i}.npy')
    # fit
    else:
        # fit broad region
        fitspec.fit_broad(spectra_broad)

        # fit li region
        fitspec.fit_li(spectra) 
        
        # get error
        fitspec.posterior(spectra) # calculates the error approx and posterior

        if args.save_fit:
            fitspec.save(f'{info_directory}/fits/{i}.npy')

    if args.plot:
        # plot broad region
        fitspec.plot_broad(spectra_broad)
        # plot Li region
        fitspec.plot_li(spectra, mode='minimize')
        if fitspec.run_res[fitspec.runs]['results'] is not None:
            # plot cornerplot
            fitspec.plot_corner()
            # plot Li region
            fitspec.plot_li(spectra, mode='posterior')
    
    if args.save:
        li_fit = fitspec.li_fit
        # if no posterior fit was done
        if li_fit is None:
            li_fit = fitspec.li_init_fit
            li_fit['err'] = [np.nan, np.nan]
            
        data_line = [i, *li_fit['amps'], li_fit['std'], li_fit['rv'], *li_fit['err'], fitspec.norris, SNR[ind], li_fit['minchisq'], fitspec.run_res[fitspec.runs]['edge_ind'], li_fit['const']]
        data.append(data_line)

if __name__ == '__main__':
    # argparse to change keys easily
    parser = argparse.ArgumentParser(description='options for running')
    parser.add_argument('-k', '--key', metavar='key', type=str, default='test', help='change the observations which are run, year/month. Needs to match id_dict contents')
    parser.add_argument('--sid', metavar='sid', type=int, default=131120002001376, help='The sobject_id of the star to be run, default star is a "quick" test case')
    parser.add_argument('--save_fit', action='store_true', help='save individual fits, 1 file per fit (all info)')
    parser.add_argument('--load_fit', action='store_true', help='load individual fits, 1 file per fit  (all info)')
    parser.add_argument('--plot', action='store_true', help='plot results')
    parser.add_argument('--save', action='store_true', help='save simplified fit results, compiled into 1 file')
    parser.add_argument('--threads', metavar='threads', type=int, default=1, help='number of threads to use.')
    args = parser.parse_args()
    load_fit = args.load_fit

    # threading save is not a good idea
    assert not ((args.threads > 1) & (args.save))

    # if this key has already been run, don't run again
    if os.path.exists(f'{output_directory}/{args.key}.npy') and args.save:
        print(f'Please remove results file for key {args.key} to re-run results')
        objectids = []
    # testing purposes
    elif args.key == 'test':
        objectids = [args.sid]
    # actual run
    else:
        objectids = np.load(f'{info_directory}/id_dict.npy', allow_pickle=True).item()[args.key]

    if args.save:
        data = []

    if args.threads == 1:
        for i in objectids:
            run(i)
    else:
        with Pool(args.threads) as p:
            list(p.map(run, objectids))
    
    if args.save and (len(data) > 0): 
        data = np.array(data)
        x = np.rec.array([
            data[:,0],
            data[:,1],
            data[:,2],
            data[:,3],
            data[:,4],
            data[:,5],
            data[:,6],
            data[:,7],
            data[:,8],
            data[:,9], 
            data[:,10], 
            data[:,11],
            data[:,12],
            data[:,13],
            data[:,14],
            data[:,15],
            data[:,16]
            ], 
            dtype=[
                ('sobject_id', int),
                ('ew_li', np.float64),
                ('CN1', np.float64),
                ('Fe', np.float64),
                ('CN2', np.float64),
                ('V/Ce', np.float64),
                ('?1', np.float64),
                ('?2', np.float64),
                ('std', np.float64),
                ('rv', np.float64),
                ('err_low', np.float64),
                ('err_upp', np.float64),
                ('norris', np.float64),
                ('snr', np.float64),
                ('minchisq', np.float64),
                ('edge_ind', np.float64),
                ('const', np.float64)
                ]
            )
        np.save(f'{output_directory}/{args.key}.npy', x)

