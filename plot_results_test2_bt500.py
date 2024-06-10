#%%
# system level
import os
from os import path
import sys


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
test_df = pd.read_csv("scores_image_sizes.csv")
test_df['scene'] = test_df['reference_folder'].str.replace('gt_', '', regex=False)
test_size = test_df.shape[0]
test_df.columns

results_full_df = pd.read_csv('results_reeval.csv')
results_full_df = results_full_df.rename(columns={'video_id': 'distorted_folder', 'pred_score': 'NeRF-DISTS-full'})
results_full_df = results_full_df[['distorted_folder', 'NeRF-DISTS-full']]
results_df = pd.read_csv('results_fin.csv')

results_df
bt500 = pd.read_csv('Test_2_iqa.csv')
bt500 = bt500.rename(columns={'DISTS': 'DISTS(IQA-PyTorch)', 'Topiq-fr': 'Topiq-fr(IQA-PyTorch)'})
bt500 = bt500[['distorted_folder', 'BT-500', 'DISTS(IQA-PyTorch)', 'Topiq-fr(IQA-PyTorch)']]
bt500
#%%
results_df = results_df.rename(columns={'video_id': 'distorted_folder', 'pred_score': 'NeRF-DISTS'})
results_df = results_df[['distorted_folder', 'NeRF-DISTS']]
test_df = pd.merge(test_df, results_full_df, on='distorted_folder')
test_df = pd.merge(test_df, results_df, on='distorted_folder')
test_df = pd.merge(test_df, bt500, on='distorted_folder')
test_df.to_csv('results_test2_pw_sf.csv')
test_df.head(3)
#%%
test_df['TEST_DIFF'] = np.abs(test_df['DISTS(IQA-PyTorch)'].values - test_df['DISTS_full_size'].values)
print(max(test_df['TEST_DIFF'].values))
test_df['DISTS_full_size_'] = test_df['DISTS_full_size'].values
test_df

print(len(test_df))
#%%
# test_df = pd.read_csv('results_combined.csv')
test_df = test_df.rename(columns={
    'NeRF-DISTS': 'Ours',
    'NeRF-DISTS-full': 'Ours(full-size)',
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

#%%
real_scene_ids = ['train', 'm60', 'playground', 'truck', 'fortress', 'horns', 'trex', 'room']
synth_scene_ids = ['ship', 'lego', 'drums', 'ficus', 'hotdog', 'materials', 'mic', 'chair']

print(test_df['scene'].unique())

syn_df = test_df[test_df['scene'].isin(synth_scene_ids)].reset_index()
tnt_df = test_df[test_df['scene'].isin(real_scene_ids)].reset_index()
#%%
test_df

# %%
def get_correlations(col, syn_df, tnt_df, test_df):
    correlations = {}
    # For each condition, unpack the dictionary returned by compute_correlations into the final dictionary
    for prefix, df in [('syn', syn_df), ('tnt', tnt_df), ('all', test_df)]:
        corr_results = compute_correlations(df[col].values, df['BT-500'])
        for corr_type in ['plcc', 'srcc', 'ktcc']:
            correlations[f'{prefix} mos {corr_type}'] = np.abs(corr_results[corr_type])
    # for prefix, df in [('syn', syn_df), ('tnt', tnt_df), ('all', test_df)]:    
    #     corr_results = compute_correlations(df[col].values, df['DMOS'])
    #     for corr_type in ['plcc', 'srcc', 'ktcc']:
    #         correlations[f'{prefix} dmos {corr_type}'] = np.abs(corr_results[corr_type])
    return correlations


# List of metrics to compute correlations for

data = []
metrics = ['Ours', 'Ours(full-size)', #'DISTS', 'DISTS_full_size', 
           'DISTS(IQA-PyTorch)', #'Topiq-fr(IQA-PyTorch)',
        #    'DISTS_full_size', 'DISTS_square', 
        #    'A-DISTS', 'A-DISTS_full_size', 'A-DISTS_square', 
       'VIF', 'MS-SSIM', 'MAD', 'PieAPP', 'WaDiQaM', 'TOPIQ-FR',
       'LPIPS(vgg)', 'LPIPS(alex)', 'SSIM', 'PSNR', 'GMSD', 'FSIMc', 'NLPD',
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
test_df
#%%
df_corr
#%%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats


# plt.style.use('seaborn-v0_8-darkgrid')
# plt.style.use('default')
width = 469.0

def set_size(width, fraction=1, subplots=(1, 1)):
    if width == 'width':
        width_pt = 469.75502
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
    "xtick.labelsize": 6,
    "ytick.labelsize": 6
}

plt.rcParams.update(tex_fonts)

from matplotlib.lines import Line2D

COLORS = plt.cm.get_cmap('tab10', 10)  # Adjust the second argument based on the number of unique scenes
scene_to_color = {scene: COLORS(i) for i, scene in enumerate(test_df['scene'].unique())}
legend_elements = [
    # *[Line2D([0], [0], marker='x', color='w', label=scene, markersize=7, markeredgecolor=scene_to_color[scene], markeredgewidth=2) for scene in test_df['scene'].unique()],
    Line2D([0], [0], marker='x', color='w', label='Synthetic Scenes', markeredgewidth=2, markersize=7, markeredgecolor='#0080bd'), # 00b238
    Line2D([0], [0], marker='o', color='w', label='Real Scenes', markersize=9, markerfacecolor='#ff7500'), # 00b238
]

# Update scatter_plot function to accept an ax parameter
def scatter_plot(ax, metric, marker_size=10):
        
    def plot_dataset(df, metric, label, marker):
        unique_scenes = df['scene'].unique()
        for scene in unique_scenes:
            scene_df = df[df['scene'] == scene].sort_values(by='BT-500')
            ax.scatter(scene_df[metric], scene_df['BT-500'], color=scene_to_color[scene], marker=marker, s=marker_size, label=label)# Draw line connecting sorted points
            # ax.plot(scene_df[metric], scene_df['BT-500'], color=scene_to_color[scene])


    # plot_dataset(syn_df, metric, 'Synthetic Scenes', marker='x')
    # plot_dataset(tnt_df, metric, 'Real Scenes', marker='o')
    # Scatter plot for synthetic data
    ax.scatter(syn_df[metric], syn_df['BT-500'], c='#0080bd', marker='x', s=marker_size, label='Synthetic Scenes')

    # Scatter plot for real data
    ax.scatter(tnt_df[metric], tnt_df['BT-500'], c='#ff7500', marker='o', s=marker_size, label='Real Scenes')



    # Labeling the plot
    ax.set_xlabel(metric)
    ax.set_ylabel('MOS')
    # ax.legend()



# Create a 2x3 grid of subplots
fig, axs = plt.subplots(6, 3, figsize=set_size(width, subplots=(6, 3)))

# Flatten the array of axes to easily iterate over it
axs = axs.flatten()

# Plot each metric on a separate subplot
for ax, metric in zip(axs, metrics):
    scatter_plot(ax, metric)

# Adjust the layout to prevent overlapping
plt.tight_layout()
# Add the custom legend to the figure
fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.05))

fig.savefig('scatter_mos_test2.pdf', format='pdf', bbox_inches='tight')
# Display the figure
plt.show()

#%%

#%%
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 8,
    "font.size": 8,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6
}

plt.rcParams.update(tex_fonts)

X = test_df['DISTS'].values.reshape(-1, 1)
Y = test_df['BT-500'].values
A = np.vstack([X.T, np.ones(len(X))]).T
m, c = np.linalg.lstsq(A, Y, rcond=None)[0]

plt.figure(figsize=set_size(90))

# Plotting scatter points
plt.scatter(test_df['DISTS'], test_df['BT-500'], marker='o', s=5, color='#0080bd')

# Plotting regression line
x_vals = np.array(plt.xlim())
y_vals = m * x_vals + c
plt.plot(x_vals, y_vals, color='#E54B4B')

# Customizing the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

plt.tick_params(
    axis='both',          # changes apply to the both axes
    which='both',      # both major and minor ticks are affected
    # bottom=False,      # ticks along the bottom edge are off
    # top=False,         # ticks along the top edge are off
    # labelbottom=True, # labels along the bottom edge are off
    # left=False,        # ticks along the left edge are off
    # right=False,       # ticks along the right edge are off
    # labelleft=True    # labels along the left edge are off
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False, # no labels along the bottom edge
    left=False,        # ticks along the left edge are off
    right=False,       # ticks along the right edge are off
    labelleft=False    # no labels along the left edge
)


plt.xlabel('DISTS')
plt.ylabel('MOS')

plt.savefig('linear_regression.pdf', format='pdf', bbox_inches='tight')

plt.show()

# %%
def get_correlations(col, syn_df, tnt_df, test_df):
    correlations = {}
    # For each condition, unpack the dictionary returned by compute_correlations into the final dictionary
    for prefix, df in [('syn', syn_df), ('tnt', tnt_df), ('all', test_df)]:
        corr_results = compute_correlations(df[col].values, df['BT-500'])
        for corr_type in ['plcc', 'srcc', 'ktcc']:
            correlations[f'{prefix} mos {corr_type}'] = np.abs(corr_results[corr_type])
    # for prefix, df in [('syn', syn_df), ('tnt', tnt_df), ('all', test_df)]:    
    #     corr_results = compute_correlations(df[col].values, df['DMOS'])
    #     for corr_type in ['plcc', 'srcc', 'ktcc']:
    #         correlations[f'{prefix} dmos {corr_type}'] = np.abs(corr_results[corr_type])
    return correlations


# List of metrics to compute correlations for
# metrics = ['Ours', 'DISTS', 'Contrique', 'GMSD', 'MS-SSIM', 'PSNR', 'LPIPS (AlexNet)', 'LPIPS (VGG)', 'WaDiQaM', 'SSIM', 'CompressVQA', 'FVVHD']
data = []

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
# %%
columns_of_interest = ['Metric', 'syn mos plcc', 'syn mos srcc', 'syn mos ktcc', 'tnt mos plcc', 'tnt mos srcc', 'tnt mos ktcc', 'all mos plcc', 'all mos srcc', 'all mos ktcc']
data_subset = df_corr[columns_of_interest]
# Rename the columns to fit LaTeX table format
# data_subset.columns = ['Metric', 'PLCC', 'SRCC', 'KRCC']

# Format the values to four decimal places
# data_subset = data_subset.round(3)

medals = ['\\goldmedal', '\\silvermedal', '\\bronzemedal']
for col in ['syn mos plcc', 'syn mos srcc', 'syn mos ktcc', 'tnt mos plcc', 'tnt mos srcc', 'tnt mos ktcc', 'all mos plcc', 'all mos srcc', 'all mos ktcc']:
    sorted_idx = data_subset[col].sort_values(ascending=False).index
    for i, medal in enumerate(medals, start=0):
        data_subset.loc[sorted_idx[i], col] = f"{float(data_subset.loc[sorted_idx[i], col]):.04f} {medal}"

    for i in range(3,len(data_subset)):
        data_subset.loc[sorted_idx[i], col] = f"{float(data_subset.loc[sorted_idx[i], col]):.04f}"

data_subset

def convert_to_latex_grouped(data_subset):
    # Starting the LaTeX table and caption
    latex_table = "\\begin{table*}[ht]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabularx}{\\textwidth}{l|X@{}X@{}X|X@{}X@{}X|X@{}X@{}X}\n"
    # latex_table += "\\begin{tabular}{l|ccc|ccc|ccc}\n"  # Adjust the number of columns based on your data
    latex_table += "\\hline \\hline\n"
    # Group headers
    latex_table += "& \\multicolumn{3}{c|}{Synthetic} & \\multicolumn{3}{c|}{Real} & \\multicolumn{3}{c}{Combined} \\\\\n"
    latex_table += "\\hline\n"
    # Sub headers
    latex_table += "\\textbf{METRICS} & PLCC & SRCC & KTCC & PLCC & SRCC & KTCC & PLCC & SRCC & KTCC \\\\\n"
    latex_table += "\\hline\n"
    
    # Adding the data rows
    for index, row in data_subset.iterrows():
        # Assuming the row contains the values in the exact order needed for the table
        # You might need to adjust how you access the row values based on the actual DataFrame structure
        latex_table += f"{row['Metric']}&{row['syn mos plcc']}&{row['syn mos srcc']}&{row['syn mos ktcc']}&{row['tnt mos plcc']}&{row['tnt mos srcc']}&{row['tnt mos ktcc']}&{row['all mos plcc']}&{row['all mos srcc']}&{row['all mos ktcc']} \\\\\n"
    
    # Ending the table
    latex_table += "\\hline \\hline\n"
    latex_table += "\\end{tabularx}\n"
    latex_table += "\\caption{Correlation results between quality assessment metrics and MOS.}\n"
    latex_table += "\\label{table:combined_mos_correlations}\n"
    latex_table += "\\end{table*}\n"
    
    return latex_table

latex_code = convert_to_latex_grouped(data_subset)

# Save the LaTeX code to a file
with open('results_table_test2.tex', 'w') as file:
    file.write(latex_code)



# %%
# df_corr.to_csv('correlations.csv')

# %%
set_size(90)
# %%
