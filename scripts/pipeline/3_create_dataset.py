%load_ext autoreload
%autoreload 2

from collections import Counter
import random

random.seed(42)

import librosa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import numpy as np
import scienceplots
import soundfile as sf

plt.style.use(['science'])

from src.utils import (
    load_json, load_pitch_track, write_pitch_track, cpath, load_annotations, get_plot_kwargs, 
    pitch_seq_to_cents, subsample_series, interpolate_below_length, smooth_pitch_curve,
    align_time_series, write_pkl, load_pkl, remove_leading_trailing_nans)

from src.pitch import transpose_pitch


svaras = ['sa', 'ri', 'ga', 'ma', 'pa', 'dha', 'ni']

min_len_prop = 0.6
data_path = 'data/interp_100/bhairavi.pkl'
out_dir = '../DeepGRU/data_interp_100/bhairavi'
pkl_path = 'data/interp_100/bhairavi_gru.pkl'
max_ran = 800
train_size = 0.8
test_size = 0.2

data = load_pkl(data_path)

tracks = data.keys()

print('Isolating pitch tracks...')
all_svaras = []
for t, d in data.items():
    if not t in tracks:
        continue

    print(f'{t}')
    annotations = d['annotations']
    pitch = d['pitch']
    time = d['time']
    timestep = d['timestep']
    beat_length = d['beat_length']
    print(f'    {len(annotations)} annotations')
    n = 0
    n2 = 0 
    n3 = 0
    n4 = 0
    for i, row in annotations.iterrows():
        start = row['start_sec']
        end = row['end_sec']
        label = row['label']
        
        if label not in svaras:
            n4 += 1
            continue

        s1_beat = start/beat_length
        s2_beat = end/beat_length
        
        s1 = round(s1_beat/timestep)
        s2 = round(s2_beat/timestep)

        psamp = pitch[s1:s2]
        psamp = remove_leading_trailing_nans(psamp)

        if len(psamp) == 0:
            n+=1
            continue
        
        if len(psamp)/len(pitch[s1:s2]) <= min_len_prop:
            n+=1
            continue

        if sum(np.isnan(psamp)>0):
            n+=1
            continue

        psamp, trans = transpose_pitch(psamp)

        if trans:
            n2 += 1

        ran = max(psamp)-min(psamp)
        
        # exclude incorrectly extracted time series
        if ran > max_ran:
            n3 += 1
            #print(f'{i}')
            continue

        all_svaras.append((svaras.index(label), psamp, i, t))

    print(f'    -> {n} excluded due to silence in pitch track')
    print(f'    -> {n3} excluded due to pitch jump errors')
    print(f'    -> {n4} excluded due to unknown label')
    print(f'    -> {n2} transposed')


counted = Counter([x[0] for x in all_svaras])
for k,v in counted.items():
    print(f"{v} instances of {svaras[k]}") 


print('Supersampling...')
n_max = max(counted.values())
for svara in range(len(svaras)):
    these_svaras = [x for x in all_svaras if x[0]==svara]
    if len(these_svaras) == 0:
        continue
    n = len(these_svaras)
    while n < n_max:
        this_svara = random.choice(these_svaras)
        all_svaras.append(this_svara)
        n += 1


counted = Counter([x[0] for x in all_svaras])
for k,v in counted.items():
    print(f"{v} instances of {svaras[k]}")



random.shuffle(all_svaras)

all_svaras_trim = [(x[0], x[1]) for x in all_svaras] # just label and sample

train_data = np.array(all_svaras_trim[:round(train_size*len(all_svaras_trim))])
test_data = np.array(all_svaras_trim[round(train_size*len(all_svaras_trim)):])

# Write
print(f'Writing to {out_dir}')
train_path = cpath(out_dir, 'TRAIN.pkl')
test_path = cpath(out_dir, 'TEST.pkl')
write_pkl(train_data, train_path)
write_pkl(test_data, test_path)

label_path = cpath(out_dir, 'LABELS.pkl')
label_lookup = dict(enumerate(svaras))
write_pkl(label_lookup, label_path)

write_pkl(all_svaras, pkl_path)