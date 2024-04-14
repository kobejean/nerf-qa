#%%
# system level
import os
from os import path
import sys

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

# deep learning
import numpy as np

# data 
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from PIL import Image

def compute_correlations(pred_scores, mos):
    plcc = pearsonr(pred_scores, mos)[0]
    srcc = spearmanr(pred_scores, mos)[0]
    ktcc = kendalltau(pred_scores, mos)[0]

    return {
        'plcc': plcc,
        'srcc': srcc,
        'ktcc': ktcc,
    }
#%%
# TEST_DATA_DIR = "/home/ccl/Datasets/NeRF-QA"

participants_df = pd.read_csv("participants.csv")
test_df = pd.read_csv("scores_aspect-3.csv")
test_df['scene'] = test_df['reference_folder'].str.replace('gt_', '', regex=False)
test_size = test_df.shape[0]
test_df.columns
results_df = pd.read_csv('results_v344.csv')

results_df
#%%
subject_ids = [f'Subject {i}' for i in range(1,48)]
participants_df = participants_df[['distorted_folder', 'method', *subject_ids]]
#%%
results_df = results_df.rename(columns={'video_id': 'distorted_folder', 'pred_score': 'NeRF-DISTS'})
results_df = results_df[['distorted_folder', 'NeRF-DISTS']]
test_df = pd.merge(test_df, results_df, on='distorted_folder')
test_df = pd.merge(test_df, participants_df, on='distorted_folder')
test_df.to_csv('results_combined_test2.csv')
test_df.head(3)
#%%
test_df.columns
#%%
# test_df = pd.read_csv('results_combined.csv')
test_df = test_df.rename(columns={
    'NeRF-DISTS': 'Ours',
    'PSNRq': 'PSNR',
    'SSIMq': 'SSIM',
    'gmsd': 'GMSD',
    'fsimc': 'FSIMc',
    'nlpd': 'NLPD',
    'mad': 'MAD',
    'Topiq-fr': 'TOPIQ-FR',
    'lpips_vgg': 'LPIPS(vgg)',
    'LPIPS(alex)': 'LPIPS(alex)',
    'Ahiq': 'AHIQ',
    'Pieapp': 'PieAPP',
    'ST-LPIPS-vgg': 'ST-LPIPS',
    'wadiqam': 'WaDiQaM'
})
# test_df['Ours'] = test_df['NeRF-DISTS'] 
test_df.columns
#%%
real_scene_ids = ['train', 'm60', 'playground', 'truck', 'fortress', 'horns', 'trex', 'room']
synth_scene_ids = ['ship', 'lego', 'drums', 'ficus', 'hotdog', 'materials', 'mic', 'chair']

print(test_df['scene'].unique())

syn_df = test_df[test_df['scene'].isin(synth_scene_ids)].reset_index()
tnt_df = test_df[test_df['scene'].isin(real_scene_ids)].reset_index()
#%%
# %%
def get_correlations(col, syn_df, tnt_df, test_df):
    correlations = {}
    # For each condition, unpack the dictionary returned by compute_correlations into the final dictionary
    for prefix, df in [('syn', syn_df), ('tnt', tnt_df), ('all', test_df)]:
        corr_results = compute_correlations(df[col].values, df['MOS'])
        for corr_type in ['plcc', 'srcc', 'ktcc']:
            correlations[f'{prefix} mos {corr_type}'] = np.abs(corr_results[corr_type])
    return correlations


# List of metrics to compute correlations for

data = []
metrics = ['Ours', 'DISTS', 'DISTS_full_size', 'DISTS_square', 'A-DISTS', 'LPIPS(alex)',
       'VIF', 'MS-SSIM', 'MAD', 'PieAPP', 'WaDiQaM', 'TOPIQ-FR',
       'LPIPS(vgg)', 'SSIM', 'PSNR', 'GMSD', 'FSIMc', 'NLPD',
       'ST-LPIPS', 'AHIQ']

# Assuming syn_df, tnt_df, and test_df are your DataFrames with the data
for metric in metrics:
    correlations = get_correlations(metric, syn_df, tnt_df, test_df)
    correlations['Metric'] = metric
    data.append(correlations)

# Creating the DataFrame
df_corr = pd.DataFrame(data)
# df_corr = df_corr.set_index('Metric')
# df_corr
#%%
df_corr
#%%




# %%
import random
data = []

def select_valid_sample(df, scenes, subject_ids):
    while True:
        scene = random.choice(scenes)
        subject = random.choice(subject_ids)
        sample_df = df[df['scene'] == scene]
        if not sample_df[[subject]].isna().all().all():
            break  # Break the loop if no NaN values are found
    sample_df = sample_df[~sample_df[[subject]].isna()]
    return scene, subject, sample_df.copy()

from tqdm import tqdm

for dataset, df in [('Synthetic', syn_df), ('Real', tnt_df)]:
    scenes = df['scene'].unique().tolist()
    methods =  df['method'].unique().tolist()
    
    n_bootstrap_samples = 10
    for _ in tqdm(range(n_bootstrap_samples)):
        n_samples = len(scenes) * len(subject_ids)
        sample_dfs = []
        for i in range(n_samples):
            scene, subject, sample_df = select_valid_sample(df, scenes, subject_ids)
            sample_df.loc[:, 'MOS_bootstrap'] = sample_df.loc[:, (subject)]
            sample_df.drop(columns=subject_ids, inplace=True)
            sample_dfs.append(sample_df)
        sample_dfs = pd.concat(sample_dfs, ignore_index=True)
        grouped_df = sample_dfs.groupby('method').mean(numeric_only=True)
        # display(grouped_df)
        # print(grouped_df.columns)
        
        for metric in metrics:
            row_data = {}
            correlations = compute_correlations(sample_dfs[metric].values, sample_dfs['MOS_bootstrap'])
            row_data['metric'] = metric
            # row_data['scene'] = scene
            row_data['dataset'] = dataset
            row_data['subjective_measure'] = 'MOS_bootstrap'
            for corr_type in ['plcc', 'srcc', 'ktcc']:
                row_data[corr_type] = np.abs(correlations[corr_type])
            data.append(row_data)
        


# Creating the DataFrame
df_corr = pd.DataFrame(data)
df_corr
#%%
df_corr[df_corr['metric'] == 'DISTS']
#%%
df_corr[df_corr['metric'] == 'DISTS_square']
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
def set_size(width, fraction=1, subplots=(1, 1)):
    if width == 'textwidth':
        width_pt = 505.89
    elif width == 'column':
        width_pt = 229.8775
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 8,
    "font.size": 8,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)
def plot_corr_distributions(ax, dataset='Synthetic', subjective_measure='MOS_bootstrap', corr_type='srcc'):
    condition = (df_corr['dataset'] == dataset) & (df_corr['subjective_measure'] == subjective_measure)
    df_plot = df_corr[condition]
    mean_correlation = df_plot.groupby('metric')[corr_type].mean().reset_index()
    mean_correlation = mean_correlation.sort_values(by=corr_type, ascending=False)
    
    sorted_metrics = mean_correlation['metric'].tolist()
    df_plot['metric'] = pd.Categorical(df_plot['metric'], categories=sorted_metrics, ordered=True)
    df_plot = df_plot.sort_values('metric')

    sns.violinplot(ax=ax, x='metric', y=corr_type, data=df_plot, inner=None, width=0.5, color='skyblue', edgecolor='k', linewidth=0.5)
    # sns.swarmplot(ax=ax, x='metric', y=corr_type, data=df_plot, size=2.0, color='k')
    sns.pointplot(ax=ax, x='metric', y=corr_type, data=df_plot, dodge=True, join=False, markers="+", color='red', scale=0.75, ci=None, order=sorted_metrics)

    for i, metric in enumerate(sorted_metrics):
        mean_val = df_plot[df_plot['metric'] == metric][corr_type].mean()
        ax.text(i, mean_val, f'{mean_val:.2f}', color='black', ha='left', va='bottom')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title(r'\textbf{' + dataset + r' Dataset}')
    ax.set_ylabel(corr_type.upper())
    ax.set_xlabel('Metric')

def plot_all_distributions(corr_type):
    # size = set_size('textwidth', subplots=(2, 1))[0]
    fig, axs = plt.subplots(2, 1, figsize=set_size('textwidth', subplots=(2, 2)))
    plot_corr_distributions(axs[0], 'Synthetic', 'MOS_bootstrap', corr_type)
    plot_corr_distributions(axs[1], 'Real', 'MOS_bootstrap', corr_type)
    plt.tight_layout()

    fig.savefig(f'violin_{corr_type}_mos.pdf', format='pdf', bbox_inches='tight')
    plt.show()
plot_all_distributions('plcc')
plot_all_distributions('srcc')
plot_all_distributions('ktcc')
# %%
