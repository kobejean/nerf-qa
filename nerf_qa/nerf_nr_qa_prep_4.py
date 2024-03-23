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
# %%
import numpy as np
def compute_std_mean(group):
    frame_count = group['frame_count'].values[0]
    basenames = eval(group['basenames'].values[0])
    render_dirs = group['render_dir'].values
    for frame in range(frame_count):
        score_maps = []
        for render_dir in render_dirs:
            if os.path.basename(render_dir) == 'color':
                score_map_dir = os.path.join(os.path.dirname(render_dir), 'score-map')
            else:
                score_map_dir = os.path.join(os.path.dirname(render_dir), 'gt-score-map')
            basename = basenames[frame]
            score_map_path = os.path.join(DATA_DIR, score_map_dir, basename)
            score_map = torch.load(score_map_path, map_location='cpu').detach().numpy()
            score_maps.append(score_map)

        score_maps = np.stack(score_maps)
        score_maps_std = np.std(score_maps, axis=0, keepdims=True)
        score_maps_std = np.broadcast_to(score_maps_std, shape=score_maps.shape)
        score_maps_mean = np.mean(score_maps, axis=0, keepdims=True)
        score_maps_mean = np.broadcast_to(score_maps_mean, shape=score_maps.shape)
        score_maps = np.concatinate([score_maps,score_maps_std,score_maps_mean], axis=1)

        for score_map, render_dir in zip(score_maps, render_dirs):
            print(score_map.shape)
            if os.path.basename(render_dir) == 'color':
                score_map_dir = os.path.join(os.path.dirname(render_dir), 'score-map')
            else:
                score_map_dir = os.path.join(os.path.dirname(render_dir), 'gt-score-map')
            basename = basenames[frame]
            score_map_basename = os.path.splitext(basename)[0] + '.pt'
            score_map_path = os.path.join(DATA_DIR, score_map_dir, score_map_basename)
            torch.save(score_map, score_map_path)

    return pd.Series({})

black_list_train_methods = [
        #'mip-splatting', 'gaussian-splatting', 'gt'
    ]
filtered_df = df[~df['method'].isin(black_list_train_methods)].reset_index()

# Group by 'scene', apply the compute_std_mean function, and reset index to flatten the DataFrame
result = filtered_df.groupby('scene').apply(compute_std_mean)
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
