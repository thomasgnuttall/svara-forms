%load_ext autoreload
%autoreload 2

import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import random
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from src.utils import write_pkl, load_pkl, cpath
from src.plotting import plot_with_context
from src.clustering import dbscan, silhouette_score, calinski_harabasz_index, davies_bouldin_index, gap_statistic, dbscan_precomp
from src.dtw import evaluate_clustering

from bad import bad

svaras = ['sa', 'ri', 'ga', 'ma', 'pa', 'dha', 'ni']

dtw_out_path = 'data/interp_100//dtw_distances.csv'
gru_data_path = 'data/interp_100/bhairavi_gru.pkl'

results_dir = cpath('results/bhairavi_interp100/gridsearch_full_dtw')
results_path = cpath(results_dir, 'gridsearch_results.csv')
gridsearch_kwargs_path = cpath(results_dir, 'gridsearch_kwargs.csv')
dtw_distances = pd.read_csv(dtw_out_path)

all_svaras = load_pkl(gru_data_path)

allowed_indices = []
tested = []
for i in range(len(all_svaras)):
    j, psamp, annot_ix, track = all_svaras[i]
    t = f'{track}_{annot_ix}'
    if t not in tested and t not in bad:
        allowed_indices.append(i)
    tested.append(t)

print('Creating distances dict')
distance_dict = {}
for i, (i1,i2,dist) in dtw_distances.iterrows():
    i1 = int(i1)
    i2 = int(i2)
    if i1 not in distance_dict:
        distance_dict[i1] = {}

    distance_dict[i1][i2] = dist


clust_data = {s:{} for s in svaras}
for svara in svaras:

    svara_ix = svaras.index(svara)

    # Get data for this svara
    ix = [i for i,x in enumerate(all_svaras) if x[0]==svara_ix and i in allowed_indices]

    clust_data[svara]['ix'] = ix


all_X = np.empty((len(all_svaras), len(all_svaras)))
all_X[:] = np.nan
for i,d in distance_dict.items():
    for j,dist in d.items():
        all_X[i,j] = dist
        all_X[j,i] = dist


print('Gridsearch...')
results_df = pd.DataFrame()
all_ms = [None] + list(np.arange(2, 20, 1))
all_mcs = list(range(2, 4))
all_norm = [True]
all_cse = [float(x) for x in np.arange(0.0, 0.04, 0.01)]
all_alphas = list(np.arange(0.1, 1.5, 0.01))
all_csm = ['leaf']
count=0
for ms in tqdm.tqdm(all_ms):
    for mcs in all_mcs:
        for csm in all_csm:
            for cse in all_cse:
                for alpha in all_alphas:
                    for norm in all_norm:
                        for svara in svaras:
                            count += 1
                            ix = np.array(clust_data[svara]['ix'])
                            kwargs =  {
                                'min_samples':ms, 
                                'min_cluster_size':mcs, 
                                'cluster_selection_method':csm,
                                'cluster_selection_epsilon': cse,
                                'alpha': alpha
                            }
                            print(kwargs)

                            X = all_X[ix, :][:, ix]
                            X = sklearn.preprocessing.normalize(X)

                            #clus_labels = dbscan(X, **kwargs)
                            clus_labels = dbscan_precomp(X, **kwargs)
                            
                            zipped = [x for x in zip(clus_labels, ix) if x[0] != -1]
                            clus_labels_real = [x[0] for x in zipped]
                            ix_real = [x[1] for x in zipped]

                            try:
                                kwargs['silhouette_score'] = silhouette_score(ix_real, clus_labels_real, distance_dict)
                            except:
                                kwargs['silhouette_score'] = np.nan

                            kwargs['svara'] = svara
                            kwargs['num_excluded'] = len([x for x in clus_labels if x == -1])
                            kwargs['num_included'] = len(clus_labels_real)
                            kwargs['num_clusters'] = len(set(clus_labels_real))

                            new_row = pd.DataFrame([kwargs])
                            results_df = pd.concat([results_df, new_row], ignore_index=True)

results_df = pd.read_csv(results_path)

metric = 'silhouette_score'
results_df_new = results_df[
    ['svara', 'silhouette_score', 'num_excluded', 'num_included', 
    'num_clusters', 'min_samples', 'min_cluster_size', 'cluster_selection_method', 'cluster_selection_epsilon', 'alpha']].sort_values(by=['svara', metric], ascending=False)
 
 # num excluded was 100
results_df_new = results_df_new[
    (results_df_new['num_clusters']>=8) &\
    #(results_df_new['num_excluded']<100) &\
    (results_df_new['num_clusters']<=20)
    ]

kwargs = {}
for svara, df in results_df_new.groupby('svara'):
    #df = df.sort_values(by='num_excluded')
    row = df.iloc[0]
    av_dist = row[metric]
    num_excluded = row['num_excluded']
    min_samples = row['min_samples']
    min_cluster_size = row['min_cluster_size']
    cluster_selection_method = row['cluster_selection_method']
    num_clusters = row['num_clusters']
    num_included = row['num_included']
    min_samples = row['min_samples']
    min_cluster_size = row['min_cluster_size']
    cluster_selection_method = row['cluster_selection_method']
    cluster_selection_epsilon = row['cluster_selection_epsilon']
    alpha = row['alpha']

    print()
    print(f'Svara, {svara}, {metric}={av_dist}, num_excluded={num_excluded}, num_included={num_included}, k={num_clusters}')
    print(f'   min_samples={min_samples}')
    print(f'   min_cluster_size={min_cluster_size}')
    print(f'   cluster_selection_method={cluster_selection_method}')
    print(f'   alpha={alpha}')
    print(f'   cluster_selection_epsilon={cluster_selection_epsilon}')

    kwargs[svara] =  {
        'min_samples':min_samples, 
        'min_cluster_size':min_cluster_size, 
        'cluster_selection_method':cluster_selection_method,
        'cluster_selection_epsilon': cluster_selection_epsilon,
        'alpha': alpha
    }


write_pkl(kwargs, gridsearch_kwargs_path)
results_df.to_csv(results_path, index=False)