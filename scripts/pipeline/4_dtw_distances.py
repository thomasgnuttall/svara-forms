%load_ext autoreload
%autoreload 2

import os

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import random
from fastdtw import fastdtw

from src.utils import write_pkl, load_pkl, cpath
#from src.dtw import dtw_path

svaras = ['sa', 'ri', 'ga', 'ma', 'pa', 'dha', 'ni']


gru_data_path = 'data/bhairavi_gru.pkl'
dtw_out_path = cpath('data/dtw_distances.csv')
plot = True
samp = None
r = 0.3
norm = False

all_svaras = load_pkl(gru_data_path)

try:
    print('Removing previous distances file')
    os.remove(dtw_out_path)
except OSError:
    pass

##text=List of strings to be written to file
header = 'index1,index2,dtw'
with open(dtw_out_path,'a') as file:
    file.write(header)
    file.write('\n')
    for i in tqdm.tqdm(list(range(len(all_svaras)))):
        for j in range(len(all_svaras)):
            jl, jpsamp, jannot_ix, jtrack = all_svaras[j]
            il, ipsamp, iannot_ix, itrack = all_svaras[i]

            if jl != il:
                continue

            if il > jl:
                continue

            l_longest = np.max([len(ipsamp), len(jpsamp)])
            radius = round(l_longest*r)
            
            if radius < 2:
                radius = 2

            # Compute DTW distance between ipsamp and jpsamp
            #path, distance = dtw_path(ipsamp, jpsamp, radius=radius, norm=norm)
            distance, path = fastdtw(ipsamp, jpsamp, dist=lambda x, y: np.linalg.norm(x - y))
            distance = distance/len(path)

            line =f"{i},{j},{distance}"
            file.write(line)
            file.write('\n')
            
            line =f"{j},{i},{distance}"
            file.write(line)
            file.write('\n')

