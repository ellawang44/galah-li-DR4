# script to split the sobject_ids apart. 
# The run results are saved at the end, so if something crashes, chunking the objects means not everything is lost
# Currently chunked on year + month, then broken down further
import os
import numpy as np
from collections import defaultdict
from config import *
from string import Template as template

alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'

def read_split():
    '''Read in sids from filename of stacked spectra. Split into year/month chunks. 
    '''

    data = []
    for short_id in os.listdir(f'{working_directory}'):
        for sid in os.listdir(f'{working_directory}/{short_id}'):
            data.append(int(sid[:-5]))
    data = np.array(list(set(data)))

    # keep 4 values, splits data into chunks of year/month
    dates = set(np.floor(data/100000000000))
    digits = len(str(list(dates)[0]).split('.')[0])

    # like this is slow, but it only takes minutes and is done once so who cares
    split_dates = defaultdict(list)
    for date in dates:
        for d in data: 
            if str(int(date)) == str(d)[:digits]:
                split_dates[str(int(date))].append(d)

    return split_dates

def chunk(split_dates, threshold=100000):
    '''Split the sids in each year/month even further.

    Parameters
    ----------
    split_dates : dict
        The dictionary containing the split sids. 
    threshold : int, optional
        The maximum size a chunk can have
    '''
    
    keys = list(split_dates.keys())
    for key in keys:
        l = len(split_dates[key])
        if l > threshold:
            # figure out the spitting
            chunks = int(np.ceil(l/threshold))
            chunk_size = int(np.ceil(l/chunks))
            inds = [chunk_size*i for i in range(chunks+1)]
            inds = zip(inds[:-1], inds[1:])
            # add to dictionary
            values = split_dates[key]  
            for i, (l, r) in enumerate(inds):
                split_dates[key+alphabet[i]] = values[l:r]
            # remove old key from dictionary 
            del split_dates[key]
    print('no. of chunks', len(split_dates.keys()))
    np.save(f'{info_directory}/id_dict.npy', split_dates, allow_pickle=True)

    return split_dates

def write_array(split_dates):
    '''Write a qsub array job.
    '''
    
    with open('qsub_template', 'r') as f:
        raw = template(f.read())
        ncpu = len(split_dates.keys())
        filled = raw.safe_substitute(ncpu=ncpu-1, main_directory=main_directory, keys=' '.join(split_dates.keys()))
    with open('qsub', 'w') as f:
        f.write(filled)

def write_loose(split_dates):
    '''Write a bunch of loose qsub jobs
    '''

    # make sure directory exists first
    if not os.path.isdir('qsub_dir'):
        os.mkdir('qsub_dir')

    for key in split_dates.keys():
        with open('qsub_loose_template', 'r') as f:
            raw = template(f.read())
            filled = raw.safe_substitute(main_directory=main_directory, key=key)
        with open(f'qsub_dir/qsub_{key}', 'w') as f:
            f.write(filled) 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='options for running')
    parser.add_argument('-t', '--threshold', metavar='threshold', type=int, default=40000, help='Maximum chunk size.')
    parser.add_argument('--loose', action='store_true', help='Save as loose qsub files, not array')
    args = parser.parse_args()

    split_dates = read_split()
    # split further through chunk
    split_dates = chunk(split_dates, threshold=args.threshold)

    # write qsub file
    if not args.loose:
        write_array(split_dates)
    else:
        write_loose(split_dates)

