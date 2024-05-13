import os
import numpy as np
from config import *

id_dict = np.load(f'{info_directory}/id_dict.npy', allow_pickle=True).item()

for key in id_dict.keys():
    os.system(f'qsub qsub_dir/qsub_{key}')

