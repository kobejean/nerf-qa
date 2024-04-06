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
 'drums_reference.mp4']
tnt_files = ['truck_reference.mp4', 'playground_reference.mp4',
 'train_reference.mp4', 'm60_reference.mp4']
print(test_df['reference_filename'].unique())

syn_df = test_df[test_df['reference_filename'].isin(syn_files)].reset_index()
tnt_df = test_df[test_df['reference_filename'].isin(tnt_files)].reset_index()
#%%
test_df.head(2)
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
legend_elements = [
    Line2D([0], [0], marker='x', color='w', label='Synthetic Scemes', markersize=7, markeredgecolor='#0080bd', markeredgewidth=2),
    Line2D([0], [0], marker='o', color='w', label='Real Scemes', markersize=9, markerfacecolor='#ff7500'), # 00b238
]
# Update scatter_plot function to accept an ax parameter
def scatter_plot(ax, metric, marker_size=20):
    # Scatter plot for synthetic data
    ax.scatter(syn_df[metric], syn_df['MOS'], c='#0080bd', marker='x', s=marker_size, label='Synthetic Scemes')

    # Scatter plot for real data
    ax.scatter(tnt_df[metric], tnt_df['MOS'], c='#ff7500', marker='o', s=marker_size, label='Real Scemes')

    # # Regression line for synthetic data (uncomment if needed)
    # slope, intercept, r_value, p_value, std_err = stats.linregress(syn_df[metric], syn_df['MOS'])
    # ax.plot(syn_df['MOS'], intercept + slope*syn_df['MOS'], 'r--', color='#0080bd')

    # # Regression line for real data (uncomment if needed)
    # slope, intercept, r_value, p_value, std_err = stats.linregress(tnt_df[metric], tnt_df['MOS'])
    # ax.plot(tnt_df['MOS'], intercept + slope*tnt_df['MOS'], 'r--', color='#ff7500')

    # Labeling the plot
    ax.set_xlabel(metric)
    ax.set_ylabel('MOS')
    # ax.legend()

# Create a 2x3 grid of subplots
fig, axs = plt.subplots(4, 3, figsize=set_size(width, subplots=(4, 3)))

# Flatten the array of axes to easily iterate over it
axs = axs.flatten()

# List of metrics to plot
metrics = ['Ours', 'DISTS', 'Contrique', 'FVVHD', 'GMSD', 'MS-SSIM', 'PSNR', 'LPIPS (AlexNet)', 'LPIPS (VGG)', 'WaDiQa', 'CompressVQA', 'SSIM']

# Plot each metric on a separate subplot
for ax, metric in zip(axs, metrics):
    scatter_plot(ax, metric)

# Adjust the layout to prevent overlapping
plt.tight_layout()
# Add the custom legend to the figure
fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.05))

fig.savefig('scatter_mos.pdf', format='pdf', bbox_inches='tight')
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
Y = test_df['MOS'].values
A = np.vstack([X.T, np.ones(len(X))]).T
m, c = np.linalg.lstsq(A, Y, rcond=None)[0]

plt.figure(figsize=set_size(90))

# Plotting scatter points
plt.scatter(test_df['DISTS'], test_df['MOS'], marker='o', s=5, color='#0080bd')

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
#%%
corr = compute_correlations(syn_df['NeRF-DISTS'], syn_df['MOS'])
print("syn NeRF-DISTS mos", corr)
corr = compute_correlations(tnt_df['NeRF-DISTS'], tnt_df['MOS'])
print("tnt NeRF-DISTS mos", corr)
corr = compute_correlations(test_df['NeRF-DISTS'], test_df['MOS'])
print("all NeRF-DISTS mos", corr)
corr = compute_correlations(syn_df['NeRF-DISTS'], syn_df['DMOS'])
print("syn NeRF-DISTS dmos", corr)
corr = compute_correlations(tnt_df['NeRF-DISTS'], tnt_df['DMOS'])
print("tnt NeRF-DISTS dmos", corr)
corr = compute_correlations(test_df['NeRF-DISTS'], test_df['DMOS'])
print("all NeRF-DISTS dmos", corr)
#%%
corr = compute_correlations(syn_df['DISTS'], syn_df['MOS'])
print("syn dists mos", corr)
corr = compute_correlations(tnt_df['DISTS'], tnt_df['MOS'])
print("tnt dists mos", corr)
corr = compute_correlations(test_df['DISTS'], test_df['MOS'])
print("all dists mos", corr)
corr = compute_correlations(syn_df['DISTS'], syn_df['DMOS'])
print("syn dists dmos", corr)
corr = compute_correlations(tnt_df['DISTS'], tnt_df['DMOS'])
print("tnt dists dmos", corr)
corr = compute_correlations(test_df['DISTS'], test_df['DMOS'])
print("all dists dmos", corr)
#%%

corr = compute_correlations(np.sqrt(syn_df['DISTS'].values), syn_df['MOS'])
print("syn dists mos", corr)
corr = compute_correlations(np.sqrt(tnt_df['DISTS'].values), tnt_df['MOS'])
print("tnt dists mos", corr)
corr = compute_correlations(np.sqrt(test_df['DISTS'].values), test_df['MOS'])
print("all dists mos", corr)
corr = compute_correlations(np.sqrt(syn_df['DISTS'].values), syn_df['DMOS'])
print("syn dists dmos", corr)
corr = compute_correlations(np.sqrt(tnt_df['DISTS'].values), tnt_df['DMOS'])
print("tnt dists dmos", corr)
corr = compute_correlations(np.sqrt(test_df['DISTS'].values), test_df['DMOS'])
print("all dists dmos", corr)
#%%

corr = compute_correlations(np.sqrt(syn_df['DISTS_tr'].values), syn_df['MOS'])
print("syn dists_tr mos", corr)
corr = compute_correlations(np.sqrt(tnt_df['DISTS_tr'].values), tnt_df['MOS'])
print("tnt dists_tr mos", corr)
corr = compute_correlations(np.sqrt(test_df['DISTS_tr'].values), test_df['MOS'])
print("all dists_tr mos", corr)
corr = compute_correlations(np.sqrt(syn_df['DISTS_tr'].values), syn_df['DMOS'])
print("syn dists_tr dmos", corr)
corr = compute_correlations(np.sqrt(tnt_df['DISTS_tr'].values), tnt_df['DMOS'])
print("tnt dists_tr dmos", corr)
corr = compute_correlations(np.sqrt(test_df['DISTS_tr'].values), test_df['DMOS'])
print("all dists dmos", corr)
#%%
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit

def plot_dists_mos_with_group_regression_b_ave(df, y_col='DISTS', x_col='MOS', group_col='reference_filename'):
    # Define a list of colors for the groups
    colors = [
        '#1f77b4',  # Mutated blue
        '#ff7f0e',  # Safety orange
        '#2ca02c',  # Cooked asparagus green
        '#d62728',  # Brick red
        '#9467bd',  # Muted purple
        '#8c564b',  # Chestnut brown
        '#e377c2',  # Raspberry yogurt pink
        '#7f7f7f',  # Middle gray
        '#bcbd22',  # Curry yellow-green
        '#17becf'   # Blue-teal
    ]

    def linear_func(x, a, b):
        return a + b * x

    # Plotting
    fig = go.Figure()

    unique_groups = df[group_col].unique()
    for i, group in enumerate(unique_groups):
        group_df = df[df[group_col] == group]
        group_x = group_df[x_col]
        group_y = group_df[y_col]
        
        # Fit the model for each group
        params, params_covariance = curve_fit(linear_func, group_x, group_y)
        
        # Predict using the fitted model for the group
        x_range = np.linspace(min(group_x), max(group_x), 400)
        y_pred = linear_func(x_range, *params)
        
        # Ensure we use a unique color for each group, cycling through the colors list if necessary
        color = colors[i % len(colors)]
        
        # Data points for the group
        fig.add_trace(go.Scatter(x=group_x, y=group_y, mode='markers', name=f'Data: {group}', marker_color=color))
        
        # Regression line for the group
        fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name=f'Regression: {group}', line=dict(color=color)))

    fig.update_layout(title=f'Linear Regression per Group between {y_col} and {x_col}',
                      xaxis_title=x_col,
                      yaxis_title=y_col)
    return fig
display(plot_dists_mos_with_group_regression_b_ave(test_df, 'DISTS', 'MOS'))
display(plot_dists_mos_with_group_regression_b_ave(test_df, 'DISTS', 'DMOS'))
# %%
def print_corr(col):
    corr = compute_correlations(np.sqrt(syn_df[col].values), syn_df['MOS'])
    print(f"syn {col} mos", corr)
    corr = compute_correlations(np.sqrt(tnt_df[col].values), tnt_df['MOS'])
    print(f"tnt {col} mos", corr)
    corr = compute_correlations(np.sqrt(test_df[col].values), test_df['MOS'])
    print(f"all {col} mos", corr)
    corr = compute_correlations(np.sqrt(syn_df[col].values), syn_df['DMOS'])
    print(f"syn {col} dmos", corr)
    corr = compute_correlations(np.sqrt(tnt_df[col].values), tnt_df['DMOS'])
    print(f"tnt {col} dmos", corr)
    corr = compute_correlations(np.sqrt(test_df[col].values), test_df['DMOS'])
    print(f"all {col} dmos", corr)

# %%
print_corr('SSIM')
# %%
print_corr('PSNR_Score')
# %%
print_corr('LPIPS_Score')
# %%
def get_correlations(col, syn_df, tnt_df, test_df):
    correlations = {}
    # For each condition, unpack the dictionary returned by compute_correlations into the final dictionary
    for prefix, df in [('syn', syn_df), ('tnt', tnt_df), ('all', test_df)]:
        corr_results = compute_correlations(df[col].values, df['MOS'])
        for corr_type in ['plcc', 'srcc', 'ktcc']:
            correlations[f'{prefix} mos {corr_type}'] = np.abs(corr_results[corr_type])
    for prefix, df in [('syn', syn_df), ('tnt', tnt_df), ('all', test_df)]:    
        corr_results = compute_correlations(df[col].values, df['DMOS'])
        for corr_type in ['plcc', 'srcc', 'ktcc']:
            correlations[f'{prefix} dmos {corr_type}'] = np.abs(corr_results[corr_type])
    return correlations


# List of metrics to compute correlations for
metrics = ['Ours', 'DISTS', 'Contrique', 'GMSD', 'MS-SSIM', 'PSNR', 'LPIPS (AlexNet)', 'LPIPS (VGG)', 'WaDiQa', 'SSIM', 'CompressVQA', 'FVVHD']
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
