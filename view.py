# diagnostic plots

import numpy as np
import matplotlib.pyplot as plt
from run import FitSpec
from read import read_spectra, read_meta, cut
from synth import fwhm_to_std
from PIL import Image
from config import *
from fit import filter_spec
import copy
from un_fitter import UNFitter

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

objectids = [140113002401159, 151231004901205, 160520002601357, 170220003101024, 210521002101384, 210916002101372, 211214000701367, 230304001601211]

#objectids = [int(i[:-4]) for i in os.listdir('data/fits') if f'{i[:-4]}.png' not in os.listdir('view')]


for i in objectids:
    print(i)
    spectra = read_spectra(i)
    if spectra is None:
        continue

    spectra = cut(spectra, 6695, 6719)
    spectra = filter_spec(spectra) 
    spectra_broad = copy.deepcopy(spectra)
    if len(spectra['wave_norm']) == 0:
        continue

    spectra = cut(spectra, 6704, 6711)

    # identify object
    ind = np.where(i==sobject_id)[0][0]

    fitspec = FitSpec(std=galah_std[ind], snr=SNR[ind], sid=i, teff=teff[ind], logg=logg[ind], feh=feh[ind])
    if os.path.exists(f'{info_directory}/fits/{i}.npy'):
        fitspec.load(f'{info_directory}/fits/{i}.npy')
    else:
        continue

    # plot all things together
    fitspec.plot_broad(spectra_broad, show=False, path=f'view_temp/{i}_broad.png')
    plt.close()
    fitspec.plot_li(spectra, mode='minimize', show=False, path=f'view_temp/{i}_init.png')
    plt.close()
    if fitspec.run_res[fitspec.runs]['results']['samples'] is not None:
        fitspec.plot_li(spectra, mode='posterior', show=False, path=f'view_temp/{i}_li.png')
        plt.close()
        fitspec.plot_corner(show=False, path=f'view_temp/{i}_corner.png')
        plt.close()

    broad = Image.open(f'view_temp/{i}_broad.png')
    init = Image.open(f'view_temp/{i}_init.png')
    if fitspec.run_res[fitspec.runs]['results']['samples'] is not None:
        li = Image.open(f'view_temp/{i}_li.png')
        corner = Image.open(f'view_temp/{i}_corner.png')

    fig = plt.figure(figsize=(12,12), constrained_layout=True)
    gs = fig.add_gridspec(ncols=3, nrows=2, height_ratios=[1,3])
    ax0 = fig.add_subplot(gs[0,0])
    ax0.imshow(broad)
    ax0.axis('off')
    ax1 = fig.add_subplot(gs[0,1])
    ax1.imshow(init)
    ax1.axis('off')
    if fitspec.run_res[fitspec.runs]['results']['samples'] is not None:
        ax2 = fig.add_subplot(gs[0,2])
        ax2.imshow(li)
        ax2.axis('off')
        ax3 = fig.add_subplot(gs[1,:])
        ax3.imshow(corner)
        ax3.axis('off')
    title = f'{fitspec.mode}'
    if not fitspec.run_res[fitspec.runs]['posterior_good']:
        title = title + ' ' + str(fitspec.edge_ind)
    plt.title(title)
    plt.savefig(f'view/{i}.png')
    plt.close()


