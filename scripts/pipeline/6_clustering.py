%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import random

from src.utils import write_pkl, load_pkl, cpath, write_list_to_file
from src.plotting import plot_with_context
from src.audio import save_audio
from src.clustering import dbscan_precomp

from bad import bad

svaras = ['sa', 'ri', 'ga', 'ma', 'pa', 'dha', 'ni']

data_path = 'data/interp_100/bhairavi.pkl'
gru_data_path = 'data/interp_100/bhairavi_gru.pkl'
output_dir = 'results/bhairavi_interp100'
kwargs_path = 'results/bhairavi_interp100/gridsearch_full_dtw/gridsearch_kwargs.csv'
clustering_results_path = cpath(output_dir, 'clusters_full', 'clustering_results.pkl')

plot = True
samp = None

all_svaras = load_pkl(gru_data_path)
data = load_pkl(data_path)
kwargs = load_pkl(kwargs_path)

allowed_indices = []
tested = []
for i in range(len(all_svaras)):
    j, psamp, annot_ix, track = all_svaras[i]
    t = f'{track}_{annot_ix}'
    if t not in tested and t not in bad:
        allowed_indices.append(i)
    tested.append(t)

clust_data = {s:{} for s in svaras}
for svara in svaras:

    svara_ix = svaras.index(svara)

    # Get data for this svara
    ix = [i for i,x in enumerate(all_svaras) if x[0]==svara_ix and i in allowed_indices]

    clust_data[svara]['ix'] = ix


print('Clustering...')
clustering = {}
for svara in svaras:
    print(f'    Clustering svara {svara}')
    
    ix = clust_data[svara]['ix']
    k = kwargs[svara]
    k['cluster_selection_epsilon'] = float(k['cluster_selection_epsilon'])
    k['alpha'] = float(k['alpha'])
    if k['min_samples'] is None or np.isnan(k['min_samples']):
        k['min_samples'] = None
    else:
        k['min_samples'] = int(k['min_samples'])
    print(f'      kwargs: {k}')

    X = all_X[ix, :][:, ix]
    X = sklearn.preprocessing.normalize(X)

    clus_labels = dbscan_precomp(X, **k)

    print(f'      n_clusters: {max(clus_labels)}')
    clustering[svara] = {'labels':clus_labels, 'ix':ix}


clustering_results = {s:{} for s in svaras}
done = {t:[] for t in data}
print('Saving results...')
for svara in svaras:
    print(f'    Svara {svara}')

    results = clustering[svara]
    
    labels = results['labels']
    all_ix = clust_data[svara]['ix']

    if samp:
        label_tracks = {}
        for track in data:
            label_tracks[track] = [i for i,x in enumerate(labels) if all_svaras[all_ix[i]][3]==track]

        sampled = []
        for track in label_tracks:
            if samp > len(label_tracks[track]):
                sampled += label_tracks[track]
            else:
                sampled += random.sample(label_tracks[track], samp)
    else:
        sampled = list(range(len(labels)))

    no_cluster = []
    for i in tqdm.tqdm(sampled):
        cluster = labels[i]
        ix = all_ix[i]
        _, psamp, annot_ix, track = all_svaras[ix]
        
        if cluster == -1:
            no_cluster += [(annot_ix, track)]
            continue

        if annot_ix in done[track]:
            continue

        track_data = data[track]

        timestep = track_data['timestep']
        annotations = track_data['annotations']
        pitch = track_data['pitch']
        time = track_data['time']
        timestep = track_data['timestep']
        beat_length = track_data['beat_length']
        plot_kwargs = track_data['plot_kwargs_cents']

        row = annotations.iloc[annot_ix]
        start = row['start_sec']
        end = row['end_sec']
        annot_svara = row['label']

        assert annot_svara == svara, 'annotations label does not match svara label'

        clustering_results[svara][ix] = {
            'track': track,
            'annot_ix': annot_ix,
            'cluster': cluster
        }

        if plot:
            plot_path = cpath(output_dir, 'clusters_full', svara, f'cluster_{cluster}', f'{track}_{annot_ix}.png')
            audio_path = cpath(output_dir, 'clusters_full', svara, f'cluster_{cluster}', f'{track}_{annot_ix}.wav')
            
            vocal = data[track]['vocal']

            plot_with_context(annotations, annot_ix, beat_length, timestep, pitch, time, plot_kwargs, plot_path)
            
            save_audio(annotations, vocal, annot_ix, audio_path)

        done[track].append(annot_ix)

    no_cluster_path = cpath(output_dir, 'clusters_full', svara, f'no_cluster.txt')
    write_list_to_file(no_cluster_path, no_cluster)

write_pkl(clustering_results, clustering_results_path)