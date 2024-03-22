#%%
import os
import csv
import torch
from tqdm import tqdm
from PIL import Image

from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS, prepare_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
import pandas as pd

DATA_DIR = "/home/ccl/Datasets/NeRF-NR-QA/"  # Specify the path to your DATA_DIR

# CSV file path
csv_file = "/home/ccl/Datasets/NeRF-NR-QA/output.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)
#%%
display(df.head(10))
df['DISTS'] = df['DISTS'].apply(eval)
# %%
import numpy as np

def compute_std_mean(group):
    # Stack the arrays to make a 2D numpy array
    stacked_arrays = np.stack(group['DISTS'].values)
    
    # Compute std and mean along the first axis (i.e., across the arrays)
    stds = np.std(stacked_arrays, axis=0)
    means = np.mean(stacked_arrays, axis=0)
    
    # Return a Series with the computed stds and means
    return pd.Series({'DISTS_std': stds.tolist(), 'DISTS_mean': means.tolist() })

black_list_train_methods = [
        'mip-splatting', 'gaussian-splatting', 'gt'
    ]
filtered_df = df[~df['method'].isin(black_list_train_methods)].reset_index()

# Group by 'scene', apply the compute_std_mean function, and reset index to flatten the DataFrame
result = filtered_df.groupby('scene').apply(compute_std_mean).reset_index()
#%%
display(result)
#%%
df = df.drop(['DISTS_std', 'DISTS_mean'], axis=1)
df_result = df.merge(result, on='scene')
display(df_result)
#%%

df_result.to_csv("/home/ccl/Datasets/NeRF-NR-QA/output.csv")
df_test = pd.read_csv("/home/ccl/Datasets/NeRF-NR-QA/output.csv")

df_test['DISTS_std'] = df_test['DISTS_std'].apply(eval)
display(df_test['DISTS_std'])
# %%
df_test['DISTS_mean'] = df_test['DISTS_mean'].apply(eval)
display(df_test['DISTS_mean'])
# %%

df_test['DISTS'] = df_test['DISTS'].apply(eval)
display(df_test['DISTS'])
#%%
dists = pd.concat([pd.Series(x) for x in df_test['DISTS']]).mean()
dists_std = pd.concat([pd.Series(x) for x in df_test['DISTS_std']]).mean()
dists_mean = pd.concat([pd.Series(x) for x in df_test['DISTS_mean']]).mean()
print(dists, dists_std, dists_mean)
dists = pd.concat([pd.Series(x) for x in df_test['DISTS']]).std()
dists_std = pd.concat([pd.Series(x) for x in df_test['DISTS_std']]).std()
dists_mean = pd.concat([pd.Series(x) for x in df_test['DISTS_mean']]).std()
print(dists, dists_std, dists_mean)
# %%
