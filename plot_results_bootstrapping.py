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
TEST_DATA_DIR = "/home/ccl/Datasets/NeRF-QA"
TEST_SCORE_FILE = path.join(TEST_DATA_DIR, "NeRF_VQA_MOS.csv")
test_df = pd.read_csv(TEST_SCORE_FILE)
test_df['scene'] = test_df['reference_filename'].str.replace('_reference.mp4', '', regex=False)
test_size = test_df.shape[0]
results_df = pd.read_csv('results.csv')

results_df
#%%
results_df = results_df.rename(columns={'video_id': 'distorted_filename', 'pred_score': 'NeRF-DISTS'})
results_df = results_df[['distorted_filename', 'NeRF-DISTS']]
test_df = pd.merge(test_df, results_df, on='distorted_filename')
test_df.head(3)
test_df.to_csv('results_combined.csv')
#%%
test_df = pd.read_csv('results_combined.csv')
test_df = test_df.rename(columns={
    'PSNR_Score': 'PSNR',
    'MS-SSIM_Score': 'MS-SSIM',
    'LPIPS_Score': 'LPIPS (AlexNet)',
    'LPIPS_Score_vgg': 'LPIPS (VGG)',
    'WaDiQa_score': 'WaDiQa',
    'WaDiQaM_score': 'WaDiQaM',
    'Contrique_score': 'Contrique'
})
test_df['Ours'] = test_df['NeRF-DISTS'] 

syn_files = ['ficus_reference.mp4', 'ship_reference.mp4',
 'drums_reference.mp4', 'lego_reference.mp4']
tnt_files = ['truck_reference.mp4', 'playground_reference.mp4',
 'train_reference.mp4', 'm60_reference.mp4']
print(test_df['reference_filename'].unique())

syn_df = test_df[test_df['reference_filename'].isin(syn_files)].reset_index()
tnt_df = test_df[test_df['reference_filename'].isin(tnt_files)].reset_index()
#%%
test_df.head(2)
#%%
test_df['scene'].unique()
#%%

# List of metrics to compute correlations for
metrics = ['Ours', 'DISTS', 'Contrique', 'GMSD', 'MS-SSIM', 'PSNR', 'LPIPS (AlexNet)', 'LPIPS (VGG)', 'WaDiQaM', 'SSIM', 'CompressVQA', 'FVVHD']
data = []
for dataset, df in [('Blender', syn_df), ('Tanks and Temples', tnt_df)]:
    scenes = df['scene'].unique()
    for scene in scenes:
        scene_df = df[df['scene'] == scene]
        for subjective_measure in ['MOS', 'DMOS']:
            for metric in metrics:
                row_data = {}
                correlations = compute_correlations(scene_df[metric].values, scene_df[subjective_measure])
                row_data['metric'] = metric
                row_data['scene'] = scene
                row_data['dataset'] = dataset
                row_data['subjective_measure'] = subjective_measure
                for corr_type in ['plcc', 'srcc', 'ktcc']:
                    row_data[corr_type] = np.abs(correlations[corr_type])
                data.append(row_data)
    


# Creating the DataFrame
df_corr = pd.DataFrame(data)
# df_corr = df_corr.set_index('Metric')
# df_corr
#%%
df_corr
#%%
df_corr[df_corr['metric'] == 'PSNR']
#%%
print(df_corr[df_corr['metric'] == 'PSNR'])
#%%
condition = (df_corr['dataset'] == 'Blender') & (df_corr['subjective_measure'] == 'MOS')
df_corr[condition]
#%%
df_plot = df_corr[condition]
print(df_plot)
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.figure(figsize=(12, 4))  # Set the figure size for better readability
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
# Create the violin plot
sns.violinplot(x='metric', y='srcc', data=df_plot, inner=None, color=".8")

# Overlay individual data points using swarmplot
sns.swarmplot(x='metric', y='srcc', data=df_plot, color='k', alpha=0.6)

# Add a point plot to show the mean values
sns.pointplot(x='metric', y='srcc', data=df_plot, dodge=True, join=False, palette="dark",
              markers="+", scale=0.75, ci=None)

# Improve the plot aesthetics
plt.xticks(rotation=45, ha='right')  # Rotate the x-axis labels for better readability
# plt.title('Distribution of SRCC Values by Metric with Mean Marker')  # Add a title
plt.ylabel('SRCC')  # Y-axis label
plt.xlabel('Metric')  # X-axis label

plt.tight_layout()  # Adjust the layout to make room for the rotated x-axis labels
plt.show()

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

def plot_corr_distributions(dataset = 'Blender', subjective_measure = 'MOS', corr_type='srcc'):
    condition = (df_corr['dataset'] == dataset) & (df_corr['subjective_measure'] == subjective_measure) 
    df_plot = df_corr[condition]
    # Calculate mean correlation for each metric
    mean_correlation = df_plot.groupby('metric')[corr_type].mean().reset_index()

    # Sort the mean_correlation by correlation values
    mean_correlation = mean_correlation.sort_values(by=corr_type, ascending=False)

    # Now, use this order for the x-axis categories in your plots
    # Seaborn will plot the data in the order it appears in the DataFrame, so we sort df_plot accordingly
    sorted_metrics = mean_correlation['metric'].tolist()
    df_plot['metric'] = pd.Categorical(df_plot['metric'], categories=sorted_metrics, ordered=True)
    df_plot = df_plot.sort_values('metric')

    # Create the violin plot with sorted metrics
    sns.violinplot(x='metric', y=corr_type, data=df_plot, inner=None, color='skyblue', edgecolor='k', linewidth=1)

    # Overlay individual data points using swarmplot
    sns.swarmplot(x='metric', y=corr_type, data=df_plot, size=2.0, color='k')

    # Add a point plot to show the mean values, using the sorted metric order
    sns.pointplot(x='metric', y=corr_type, data=df_plot, dodge=True, join=False,
                markers="+", color='red', scale=0.75, ci=None, order=sorted_metrics)
    
    for i, metric in enumerate(sorted_metrics):
        # Find the mean value for the current metric
        mean_val = df_plot[df_plot['metric'] == metric][corr_type].mean()
        # Place the text label right next to the point
        plt.text(i, mean_val, f'{mean_val:.2f}', color='black', ha='left', va='bottom')

    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
    plt.title(dataset)  # Title
    plt.ylabel('SRCC')  # Y-axis label
    plt.xlabel('Metric')  # X-axis label

    plt.tight_layout()  # Adjust layout
    plt.show()

plot_corr_distributions('Blender', 'MOS')
plot_corr_distributions('Tanks and Temples', 'MOS')
# plot_corr_distributions('Blender', 'DMOS')
# plot_corr_distributions('Tanks and Temples', 'DMOS')
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
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)
def plot_corr_distributions(ax, dataset='Blender', subjective_measure='MOS', corr_type='srcc'):
    condition = (df_corr['dataset'] == dataset) & (df_corr['subjective_measure'] == subjective_measure)
    df_plot = df_corr[condition]
    mean_correlation = df_plot.groupby('metric')[corr_type].mean().reset_index()
    mean_correlation = mean_correlation.sort_values(by=corr_type, ascending=False)
    
    sorted_metrics = mean_correlation['metric'].tolist()
    df_plot['metric'] = pd.Categorical(df_plot['metric'], categories=sorted_metrics, ordered=True)
    df_plot = df_plot.sort_values('metric')

    sns.violinplot(ax=ax, x='metric', y=corr_type, data=df_plot, inner=None, width=0.5, color='skyblue', edgecolor='k', linewidth=0.5)
    sns.swarmplot(ax=ax, x='metric', y=corr_type, data=df_plot, size=2.0, color='k')
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
    plot_corr_distributions(axs[0], 'Blender', 'MOS', corr_type)
    plot_corr_distributions(axs[1], 'Tanks and Temples', 'MOS', corr_type)
    plt.tight_layout()

    fig.savefig(f'violin_{corr_type}_mos.pdf', format='pdf', bbox_inches='tight')
    plt.show()
plot_all_distributions('plcc')
plot_all_distributions('srcc')
plot_all_distributions('ktcc')
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
        latex_table += f"{row['Metric']}&{row['syn mos plcc']}&{row['syn mos srcc']}&{row['syn mos ktcc']}&{row['tnt mos plcc']}&{row['tnt mos srcc']}&{row['tnt mos ktcc']}&{row['all mos plcc']}&{row['syn mos srcc']}&{row['syn mos ktcc']} \\\\\n"
    
    # Ending the table
    latex_table += "\\hline \\hline\n"
    latex_table += "\\end{tabularx}\n"
    latex_table += "\\caption{Correlation results between quality assessment metrics and MOS.}\n"
    latex_table += "\\label{table:combined_mos_correlations}\n"
    latex_table += "\\end{table*}\n"
    
    return latex_table

latex_code = convert_to_latex_grouped(data_subset)

# Save the LaTeX code to a file
with open('results_table.tex', 'w') as file:
    file.write(latex_code)



# %%
# df_corr.to_csv('correlations.csv')

# %%
set_size(90)
# %%
