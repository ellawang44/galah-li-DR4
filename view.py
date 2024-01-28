# diagnostic plots

import numpy as np
import matplotlib.pyplot as plt
from run import FitSpec
from read import read_spectra, read_meta, cut
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
vbroad = data['vbroad']
e_vbroad = data['e_vbroad']
factor = 6707.814/(2*np.sqrt(2*np.log(2)))/299792.458 # convert from km/s to \AA for std, might have a FWHM too

objectids = []
objectids.extend([170121002201384, 170121002201396, 170104002901059, 170108002201155, 170508006401346, 140812004401119, 160418005601330, 180621002901320, 160519005201183, 160520002601357, 150601003201221, 160130003101273]) # a range of different types of spectra
objectids.extend([131216002601003, 131220004401099, 140111004101214, 140708002201306, 140708003401078, 140708004701078, 140708005801203, 140708006401203, 140709001901194, 140709005801117, 140710006601104, 140710007601032, 140710008301032, 140710009001032, 140713003901146, 140806005301134, 140807001701174, 140807002101174, 140807002501174, 140807003101174, 141102003801353, 141102004001353, 150210005801171, 150607003601353]) # benchmark stars
objectids.extend([131216003201003]) # metal poor stars
objectids.extend([140314002601106, 140607000701111, 160403002501179]) # saturated stars
objectids.append(161116002201392) # high feh, giant, check blending lines
objectids.extend([180604003701205, 180604003701233]) # SNR > 1200, check blending lines
objectids.extend([170506003401371, 160403002501179]) # paper examples
objectids.append(170510005801366) # poorly constrained, 1 sigma detection
objectids.append(140314005201392) # TiO & CN filled star

for i in objectids:
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
    if np.isnan(e_vbroad[ind]):
        e_vbroad[ind] = 10
    stdu = np.sqrt(vbroad[ind]**2 + (299792.458/22000)**2)*factor # max std based on R=22000
    stdue = np.sqrt((vbroad[ind]+3*e_vbroad[ind])**2 + (299792.458/22000)**2)*factor # max std based on R=22000
    stdl = 0.09 #10km/s FWHM based on intrinsic broadening 
    #stdl = np.sqrt(vbroad[ind]**2 + (299792.458/32000)**2)*factor # min std based on R=32000
    std_galah = np.sqrt(vbroad[ind]**2 + (299792.458/25500)**2)*factor # eyeballed at R=25500
    if np.isnan(std_galah): # some DR3 have nan std
        continue

    fitspec = FitSpec(std_galah=std_galah, stdl=stdl, stdu=stdu, stdue=stdue, e_vbroad=e_vbroad[ind]*factor, snr=SNR[ind], sid=i, teff=teff[ind], logg=logg[ind], feh=feh[ind])
    if os.path.exists(f'{info_directory}/fits/{i}.npy'):
        fitspec.load(f'{info_directory}/fits/{i}.npy')
    else:
        continue

    # plot all things together
    fitspec.plot_broad(spectra_broad, show=False, path=f'view_temp/{i}_broad.png')
    plt.close()
    fitspec.plot_li(spectra, mode='minimize', show=False, path=f'view_temp/{i}_init.png')
    plt.close()
    if fitspec.sample is not None:
        fitspec.plot_li(spectra, mode='posterior', show=False, path=f'view_temp/{i}_li.png')
        plt.close()
        fitspec.plot_corner(show=False, path=f'view_temp/{i}_corner.png')
        plt.close()

    broad = Image.open(f'view_temp/{i}_broad.png')
    init = Image.open(f'view_temp/{i}_init.png')
    if fitspec.sample is not None:
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
    if fitspec.sample is not None:
        ax2 = fig.add_subplot(gs[0,2])
        ax2.imshow(li)
        ax2.axis('off')
        ax3 = fig.add_subplot(gs[1,:])
        ax3.imshow(corner)
        ax3.axis('off')
    title = f'{fitspec.mode}'
    if not fitspec.posterior_good:
        title = title + ' ' + str(fitspec.edge_ind)
    plt.title(title)
    plt.savefig(f'view/{i}.png')
    plt.close()


