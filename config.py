import os

# all the paths

# path to the spectra fits files. Only need the ones with CCD3, but it's fine if others are there as well
working_directory = '/g/data1a/y89/xw5841/galah_spec'

# directory which has access to the code, things will be launched from here
main_directory = '/g/data/y89/xw5841/galah-li-DR4'

# the intermediate directory to dump files needed to run the code 
info_directory = '/g/data/y89/xw5841/galah-li-DR4/data'
if not os.path.isdir(f'{info_directory}'):
    os.mkdir(f'{info_directory}')
# need to have a folder named fits inside it
if not os.path.isdir(f'{info_directory}/fits'):
    os.mkdir(f'{info_directory}/fits')

# info_directory also contains the GALAH DR file -- needed for sp
DR = 'galah_dr4_allstar_240207.fits'

# results directory
output_directory = '/g/data/y89/xw5841/galah-li-DR4/results'

# breidablik model directory 
model_path = '/priv/avatar/ellawang/galah-li/model'
