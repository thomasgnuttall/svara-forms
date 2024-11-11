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
from src.plotting import save_grouped_bar_chart, save_boxplot_2

svaras = ['sa', 'ri', 'ga', 'ma', 'pa', 'dha', 'ni']

data_path = 'data/interp_100/bhairavi.pkl'
gru_data_path = 'data/interp_100/bhairavi_gru.pkl'
output_dir = 'results/final'
clustering_results_path = f'{output_dir}/clusters_full/clustering_results.pkl'
context_data_path = 'data/bhairavi_context.pkl'
analysis_result_path = f'{output_dir}/analysis/'
mutual_info_path = cpath(analysis_result_path, 'mutual_info.csv')
boxplot_path = cpath(analysis_result_path, 'NMI_boxplot.png')


all_svaras = load_pkl(gru_data_path)
data = load_pkl(data_path)
clustering_results = load_pkl(clustering_results_path)
context_data = load_pkl(context_data_path)

K = 5

n_permutations = 10000
results = pd.DataFrame()
for svara in set([x[0] for x in unique_clusters.keys()]):
    print(svara)
    this_unique_clusters = {k:v for k,v in unique_clusters.items() if k[0]==svara}  

    #all_groups = [[x[0] for x in tu['sequences']] for tu in this_unique_clusters.values()]
    #all_groups = [x for y in all_groups for x in y]

    groups = []
    for group_name, group_d in this_unique_clusters.items():
        #print(group_name)
        this_svara = group_name[0]
        group = [x[0] for x in group_d['sequences']]
        groups.append(group)

    for k in tqdm.tqdm(list(range(1, K+1))):

        prec_res = permutation_test(groups, k, 'prec', n_permutations=n_permutations)
        succ_res = permutation_test(groups, k, 'succ', n_permutations=n_permutations)
        both_res = permutation_test(groups, k, 'both', n_permutations=n_permutations)

        prec_res['svara'] = svara
        succ_res['svara'] = svara
        both_res['svara'] = svara

        results = append_row(results, prec_res)
        results = append_row(results, succ_res)
        results = append_row(results, both_res)



from src.stats import cramers_v, combine_p_values, kruskal_with_epsilon_squared
from scipy.stats import kruskal

from scipy.stats import kruskal

def calculate_kruskal_results(categorical_values, continuous_values):
    """
    Calculate the epsilon squared (effect size) and p-value for a Kruskal-Wallis test 
    between a categorical variable and continuous values.

    Parameters:
    - categorical_values (list): A list of categorical group names corresponding to each 
                                  observation in the continuous data.
    - continuous_values (list): A list of continuous feature values corresponding to each 
                                observation in the categorical data.

    Returns:
    - result (dict): A dictionary containing the epsilon squared and p-value for the test.
    """
    # Group the continuous values by the categorical feature values
    grouped_data = {}
    for cat, val in zip(categorical_values, continuous_values):
        if cat not in grouped_data:
            grouped_data[cat] = []
        grouped_data[cat].append(val)
    
    # Perform Kruskal-Wallis test
    groups = list(grouped_data.values())
    H_stat, p_value = kruskal(*groups)
    
    # Calculate epsilon squared
    n = len(continuous_values)  # total sample size
    k = len(grouped_data)       # number of groups (unique categorical values)
    epsilon_squared = (H_stat - k + 1) / (n - k) if n > k else None  # Avoid division by zero
    
    # Return the results as a dictionary
    result = {
        'Epsilon Squared': epsilon_squared,
        'p-value': p_value
    }
    
    return result

print('Computing Kruskal-Wallis')
dur_res = []
for svara in svaras:
    clust = clustering_results[svara].values()
    clus_labels = [x['cluster'] for x in clust]

    all_durations = []

    for c in clust:
        track = c['track']
        annot_ix = c['annot_ix']
        beat_length = data[track]['beat_length']
        row = data[track]['annotations'].iloc[annot_ix]
        duration = (row['end_sec'] - row['start_sec'])/beat_length
        all_durations.append(duration)

    #h_stat, p_value, epsilon_squared = kruskal_with_epsilon_squared(clus_labels, all_durations)
    results = calculate_kruskal_results(clus_labels, all_durations)

    if p < 0.001:
        dur_res.append((results['Epsilon Squared'], results['p-value'], len(set(clus_labels))))
    else:
        dur_res.append((np.nan, np.nan))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def plot_effect_size_barplot(test_results, labels, filename="test.png"):
    # Extract the effect sizes from the test results
    effect_sizes = [result[0] for result in test_results]

    # Calculate the mean effect size
    mean_effect_size = np.mean(effect_sizes)

    # Create the barplot
    plt.figure(figsize=(10, 3))
    bars = plt.bar(labels, effect_sizes, color='skyblue', edgecolor='black')

    # Add a dashed line for the mean effect size
    plt.axhline(y=mean_effect_size, color='red', linestyle='--', label=f'Mean: {mean_effect_size:.2f}')

    # Set the title and labels
    plt.title('Relationship between Duration and Svara-Form Group Label', fontsize=16)
    plt.xlabel('Svara', fontsize=14)
    plt.ylabel('Effect Size \n(Epsilon squared for Kruskal-Wallis)', fontsize=14)

    # Add the mean line label
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.grid()

    # Save the plot to a file
    plt.savefig(filename)
    plt.close()

# Example usage:
plot_effect_size_barplot(dur_res, svaras)












alpha = 0.001

print("Generating Boxplot")
names = [f'prec_{i}' for i in range(1,K+1)] + [f'succ_{i}' for i in range(1,K+1)] + [f'both_{i}' for i in range(1,K+1)]
groups = []
for name in names:
    k = int(name.split('_')[-1])
    direc = name.split('_')[0]

    this = results[(results['k']==k) & (results['direction']==direc)]
    
    #this[this['p']<alpha]['metric'] = np.nan

    this['metric'] = this.apply(lambda y: np.nan if y['p'] > alpha else y['metric'], axis=1)

    this['svara'] = pd.Categorical(this['svara'], categories=[x[0] for x in svaras], ordered=True)

    this = this.sort_values(by='svara')
    
    groups.append(list(this['metric'].values))
label = [
    "Preceeding context (Cohen's d)" if 'prec' in x else \
    "Succeeding context (Cohen's d)" if 'succ' in x else \
    "Full context (Cohen's d)" if 'both' in x else \
    "Duration (Kruskal-Wallis)" for x in names]

save_boxplot_2(
    groups, names, label, 'test.png', marker_labels=svaras,
    figsize=(15,4.5), ylabel='Normalised Mutual Information',
    xlabel='Feature', 
    title='Effect Size Between Cluster Allocation and Context Feature Across All Svaras')


real_values = []
randomized_values = []
random_stds = []
effect_sizes = []
xnames = [x.replace('_',' ') for x in names]
titles = []
for svara in svaras+['sa']:
    svara2 = svara[0]
    titles.append(svara)
    this = results[(results['svara']==svara2)]
    this_real = []
    this_random = []
    this_std = []
    this_effect = []
    for name in names:
        k = int(name.split('_')[-1])
        direc = name.split('_')[0]

        that = this[(this['k']==k) & (this['direction']==direc)]
        
        p = that['p'].iloc[0]
        if p < alpha:
            real = that['metric'].iloc[0]
            rand = that['random_metric'].iloc[0]
            std = that['random_metric_std'].iloc[0]
            effect = that['effect_size'].iloc[0]
        else:
            real = np.nan
            rand = np.nan
            std = np.nan
            effect = np.nan

        this_real.append(real)
        this_random.append(rand)
        this_std.append(std)
        this_effect.append(effect)
    real_values.append(this_real)
    randomized_values.append(this_random)
    random_stds.append(this_std)
    effect_sizes.append(this_effect)
ylabels = ['Normalised Mutual Information']*len(real_values)
xlabels = ['']*len(real_values)









import matplotlib.patches as patches

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def plot_trends_with_errorbars(real_values, randomized_values, random_stds, xnames, titles, xlabels, ylabels, filepath):

    # Set up a 2x4 grid of subplots with shared X and Y axes
    fig, axes = plt.subplots(2, 4, figsize=(20, 5), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Define colors for real and randomized data points
    real_color = ['darkgreen','darkorange','darkgrey']
    random_color = 'darkgrey'
    rectangle_colors = ['lightblue', 'lightgreen', 'lightcoral']

    y_max = max([x for y in real_values for x in y])
    y_min = min([x for y in real_values for x in y])
    # Create each plot
    for i in range(8):
        ax = axes[i]
        
        # Plot real values with breaks in the line
        for I,j in enumerate(range(0, 15, 5)):  # Loop in chunks of 5
            ax.plot(xnames[j:j+5], real_values[i][j:j+5], color=real_color[0], marker='o', linestyle='-', label='Real Values' if j == 0 else "")

        ## Plot randomized values with error bars and breaks in the line
        for j in range(0, 15, 5):  # Loop in chunks of 5
            ax.errorbar(xnames[j:j+5], randomized_values[i][j:j+5], yerr=random_stds[i][j:j+5], 
                        color=random_color, marker='o', linestyle='--', label='Randomized Values' if j == 0 else "")

        # Set title, labels, and legend
        ax.set_title(titles[i], fontsize=12)
        ax.set_xlabel(xlabels[i], fontsize=10)
        ax.set_ylabel(ylabels[i], fontsize=10)
        ax.set_xticklabels(xnames, rotation=45, ha='right')
        ax.tick_params(axis='x', which='both', bottom=True, top=False)  # Show x-ticks at the bottom

        #ax.legend(loc='upper right')
        ax.grid()

    # Adjust layout and save the figure
    #plt.xticks(rotation=45, ha='right')
    
    #fig.suptitle("All Svaras", fontsize=16)
    plt.subplots_adjust(top=0.95)  # Space for the suptitle
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)


plot_trends_with_errorbars(real_values, randomized_values, random_stds, xnames, titles, xlabels, ylabels, 'test_nocount.png')
















#label = [
#    "Preceeding context (Cohen's d)" if 'prec' in x else \
#    "Succeeding context (Cohen's d)" if 'succ' in x else \
#    "Full context (Cohen's d)" if 'both' in x else \
#    "Duration (Kruskal-Wallis)" for x in names]

#boxplot_path = cpath(analysis_result_path, f'boxplot_NMI2.png')
##save_boxplot(
##    groups, names, label, boxplot_path, 
##    figsize=(15,4.5), ylabel='Effect Size',
##    xlabel='Feature', 
##    title='Effect Size Between Cluster Allocation and Context Feature Across All Svaras')


##results.to_csv(mutual_info_path, index=False)
#save_boxplot_2(
#    groups, names, label, boxplot_path, marker_labels=svaras,
#    figsize=(15,4.5), ylabel='Normalised Mutual Information',
#    xlabel='Feature', 
#    title='Effect Size Between Cluster Allocation and Context Feature Across All Svaras')
