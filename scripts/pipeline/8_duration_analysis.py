%load_ext autoreload
%autoreload 2

from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import random

from scipy.stats import chi2_contingency
from math import sqrt

from src.utils import write_pkl, load_pkl, cpath, append_row
from src.plotting import save_grouped_bar_chart, save_boxplot

svaras = ['sa', 'ri', 'ga', 'ma', 'pa', 'dha', 'ni']

data_path = 'data/interp_100/bhairavi.pkl'
gru_data_path = 'data/interp_100/bhairavi_gru.pkl'
output_dir = 'results/final'
clustering_results_path = f'{output_dir}/clusters_full/clustering_results.pkl'
context_data_path = 'data/bhairavi_context.pkl'
analysis_result_path = f'{output_dir}/analysis/'
mutual_info_path = cpath(analysis_result_path, 'mutual_info.csv')


all_svaras = load_pkl(gru_data_path)
data = load_pkl(data_path)
clustering_results = load_pkl(clustering_results_path)
context_data = load_pkl(context_data_path)


all_sequences = [x['sequences'] for x in unique_clusters.values()]
all_sequences = [y for x in all_sequences for y in x]
# cluster: {patterns: annot_ix, performances, svara, cluster}

kamakshi_unique = []
raksha_unique = []
both = []
for clus_name, clus_dict in unique_clusters.items():
    sequences = clus_dict['sequences']
    perfs = [x[1] for x in sequences]

    if set(perfs) == {'kamakshi'}:
        kamakshi_unique.append(clus_name)
    elif set(perfs) == {'raksha_bettare'}:
        raksha_unique.append(clus_name)
    else:
        both.append(clus_name)


def findstem(ss, length):
    
    def substrings(s):
        all_ss = {s[i:j]
            for j in range(len(s)+1)
            for i in range(j+1)}
        return set([x for x in all_ss if any([y.isupper() for y in x]) and len(x)==length])

    common = set.intersection(*map(substrings, ss))
    if not common:
        return None
    return max(common, key=len)

from collections import defaultdict

# Provided function to find longest common subsequence of a specified length
def findstem(ss, length):
    def substrings(s):
        all_ss = {s[i:j] for j in range(len(s)+1) for i in range(j+1)}
        return set([x for x in all_ss if any(y.isupper() for y in x) and len(x) == length])

    common = set.intersection(*map(substrings, ss))
    if not common:
        return None
    return max(common, key=len)

def group_by_common_subsequence(ss_list, length):
    # Helper function to identify the longest common subsequence in a list of strings
    def findstem(ss, length):
        def substrings(s):
            all_ss = {s[i:j] for j in range(len(s)+1) for i in range(j+1)}
            return set([x for x in all_ss if any([y.isupper() for y in x]) and len(x) == length])

        common = set.intersection(*map(substrings, ss))
        if not common:
            return None
        return max(common, key=len)

    # To store subsequences and their corresponding parent sequences
    subsequence_groups = {}

    # Iterate over each string in the list
    for i in range(len(ss_list)):
        for j in range(i + 1, len(ss_list)):
            # Get the common subsequence between the two strings
            common_subsequence = findstem([ss_list[i], ss_list[j]], length)
            if common_subsequence:
                # If a common subsequence is found, group the sequences
                if common_subsequence not in subsequence_groups:
                    subsequence_groups[common_subsequence] = []
                # Add both sequences to the subsequence group
                if ss_list[i] not in subsequence_groups[common_subsequence]:
                    subsequence_groups[common_subsequence].append(ss_list[i])
                if ss_list[j] not in subsequence_groups[common_subsequence]:
                    subsequence_groups[common_subsequence].append(ss_list[j])

    # Return the subsequences and their corresponding parent sequences
    return subsequence_groups


# This suggests that it is not just the sequence that is imporant, 
# other factors could be involved such as duration and performance 
# style or even the rhythm of the melody at that stage

length = 5
for k,d in unique_clusters.items():
    seqs = [x[0] for x in d['sequences']]
    #common = findstem(seqs, length)
    d['long_ngram_groups'] = group_by_common_subsequence(seqs, length)




to_compare = []
#groups to compare
count = 0
skipped = 0
for k in kamakshi_unique:
    print('')
    print('')
    print(f'Cluster {k} does not appear in Raksha Bettare')
    print(len(f'Cluster {k} does not appear in Raksha Bettare')*'_')
    clus_dict = unique_clusters[k]
    
    long_context = clus_dict['long_ngram_groups']
    N = len(clus_dict['sequences'])
    print(f'   ->This cluster consists of {N} svaras in the following contexts...')
    for s,_,ai in clus_dict['sequences']:
        print(f'       {s} ({ai})')

    if long_context:
        print('   ->Contexts')
    yes = 0
    for s,c in long_context.items():
        if not s:
            print('       No common context')
            skipped+=1
            continue
        n = len(c)
        X = [x[2] for x in all_sequences if s in x[0] if x[1]=='raksha_bettare']
        x = len(X)
        if x > 0:
            yes = 1
        print(f'       {n} exist in the context {s}, this subsequence appears {x} times in Raksha Bettare')
        print(f'           ids: {X}')

        X = [x for x in all_sequences if s in x[0] if x[1]=='raksha_bettare']
        Y = [x for x in all_sequences if s in x[0] if x[1]=='kamakshi']
        to_compare.append((X,Y))





    count += yes



for k in raksha_unique:
    print('')
    print('')
    print(f'Cluster {k} does not appear in Kamakshi')
    print(len(f'Cluster {k} does not appear in Kamakshi')*'_')
    clus_dict = unique_clusters[k]
    short_context = clus_dict['ngram_groups']
    long_context = clus_dict['long_ngram_groups']
    N = len(clus_dict['sequences'])
    print(f'   ->This cluster consists of {N} svaras in the following contexts...')
    for s,_,ai in clus_dict['sequences']:
        print(f'       {s} ({ai})')

    if long_context:
        print('   ->Contexts')
    yes = 0
    for s,c in long_context.items():
        if not s:
            print('       No common context')
            skipped+=1
            continue
        n = len(c)
        X = [x[2] for x in all_sequences if s in x[0] if x[1]=='kamakshi']
        x = len(X)
        if x > 0:
            yes = 1
        print(f'       {n} exist in the context {s}, this subsequence appears {x} times in Kamakshi')
        print(f'           ids: {X}')
        
        X = [x for x in all_sequences if s in x[0] if x[1]=='raksha_bettare']
        Y = [x for x in all_sequences if s in x[0] if x[1]=='kamakshi']

        to_compare.append((X,Y))

    count += yes


print(f'{count}/{len(kamakshi_unique) + len(raksha_unique)} share context but not svara-form')



for k in unique_clusters:
    print('')
    print('')
    print(f'Cluster {k}')
    print(len(f'Cluster {k}')*'_')
    clus_dict = unique_clusters[k]
    short_context = clus_dict['ngram_groups']
    long_context = clus_dict['long_ngram_groups']
    N = len(clus_dict['sequences'])
    print(f'   ->This cluster consists of {N} svaras in the following contexts...')
    for s,_,ai in clus_dict['sequences']:
        print(f'       {s} ({ai})')

    if long_context:
        print('   ->Contexts')
    yes = 0
    for s,c in long_context.items():
        if not s:
            print('       No common context')
            skipped+=1
            continue
        n = len(c)
        X = [x[2] for x in all_sequences if s in x[0] if x[1]=='kamakshi']
        x = len(X)
        if x > 0:
            yes = 1
        print(f'       {n} exist in the context {s}')
        #print(f'           ids: {X}')
        
        X = [x for x in all_sequences if s in x[0] if x[1]=='raksha_bettare']
        Y = [x for x in all_sequences if s in x[0] if x[1]=='kamakshi']

        to_compare.append((X,Y))

    count += yes





all_clustering_results = [x for y in clustering_results.values() for x in y.values()]

diffs_real = []
diffs_same = []
svara_labels_real = []
svara_labels_same = []
for seq2 in tqdm.tqdm(all_sequences):
    track2 = seq2[1]
    annot_ix2 = seq2[2]
    pat2 = seq2[0]
    cluster2 = [x for x in all_clustering_results if x['track']==track2 and x['annot_ix']==annot_ix2][0]['cluster']

    row2 = data[track2]['annotations'].iloc[annot_ix2]
    svara2 = row2['label']
    bl2 = data[track2]['beat_length']
    dur2 = (row2['end_sec'] - row2['start_sec'])/beat_length

    for seq1 in all_sequences:
        track1 = seq1[1]
        annot_ix1 = seq1[2]
        pat1 = seq1[0]
        cluster1 = [x for x in all_clustering_results if x['track']==track1 and x['annot_ix']==annot_ix1][0]['cluster']
        row1 = data[track1]['annotations'].iloc[annot_ix1]
        svara1 = row1['label']
        bl1 = data[track1]['beat_length']
        dur1 = (row1['end_sec'] - row1['start_sec'])/beat_length

        ss = findstem([pat1, pat2], 5)

        # same context, different cluster
        if ss and cluster1 != cluster2 and svara1==svara2:
            #diffs_real.append(max([dur1,dur2])/min([dur1,dur2]))
            diffs_real.append(abs(dur1-dur2))
            svara_labels_real.append(svara1)

        # same context, same cluster
        elif ss and cluster1 == cluster2 and svara1==svara2:
            #diffs_same.append(max([dur1,dur2])/min([dur1,dur2]))
            diffs_same.append(abs(dur1-dur2))
            svara_labels_same.append(svara1)





mannwhitneyu(diffs_real, diffs_same, alternative='greater')

ttest_ind(diffs_real, diffs_same)




