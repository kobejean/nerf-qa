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
csv_file = "/home/ccl/Datasets/NeRF-NR-QA/output_ADISTS.csv"

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
    score_map_log_mins = group['score_map_log_min'].apply(eval)
    score_map_log_maxs = group['score_map_log_max'].apply(eval)
    score_map_log_std_maxs = []
    score_map_log_std_mins = []
    score_map_log_mean_maxs = []
    score_map_log_mean_mins = []
    print(group['scene'].unique())
    for frame in tqdm(range(frame_count)):
        score_maps = []
        for render_dir, score_map_log_min, score_map_log_max in zip(render_dirs, score_map_log_mins, score_map_log_maxs):
            log_min = score_map_log_min[frame]
            log_max = score_map_log_max[frame]
            if os.path.basename(render_dir) == 'color':
                score_map_dir = os.path.join(os.path.dirname(render_dir), 'score-map')
            else:
                score_map_dir = os.path.join(os.path.dirname(render_dir), 'gt-score-map')
            basename = basenames[frame]
            score_map_path = os.path.join(DATA_DIR, score_map_dir, basename)
            score_map = Image.open(score_map_path)
            score_map = torch.from_numpy(np.array(score_map)).unsqueeze(0).double() / 255.0
            assert score_map.shape[0] == 1
            score_map = (log_max-log_min) * score_map + log_min
            score_map = torch.pow(10, score_map)
            score_maps.append(score_map)

        score_maps = torch.stack(score_maps)
        score_maps_std = torch.std(score_maps, dim=0, keepdim=True)
        score_maps_mean = torch.mean(score_maps, dim=0, keepdim=True)
        #score_maps = torch.concat([score_maps, score_maps_std, score_maps_mean], dim=1)
        
        score_maps_min = score_maps.amin(dim=[2,3]).squeeze(1)
        score_maps_max = score_maps.amax(dim=[2,3]).squeeze(1)
        score_log_max = (-torch.log10(score_maps_min)).numpy().tolist()
        score_log_min = (-torch.log10(score_maps_max)).numpy().tolist()

        score_maps_std_min = score_maps_std.amin(dim=[2,3]).squeeze()
        score_maps_std_max = score_maps_std.amax(dim=[2,3]).squeeze()
        score_maps_mean_min = score_maps_mean.amin(dim=[2,3]).squeeze()
        score_maps_mean_max = score_maps_mean.amax(dim=[2,3]).squeeze()
        score_log_std_max = (-torch.log10(score_maps_std_min))
        score_log_std_min = (-torch.log10(score_maps_std_max))
        score_log_mean_max = (-torch.log10(score_maps_mean_min))
        score_log_mean_min = (-torch.log10(score_maps_mean_max))

        score_map_log_std_maxs.append(score_log_std_max.item())
        score_map_log_std_mins.append(score_log_std_min.item())
        score_map_log_mean_maxs.append(score_log_mean_max.item())
        score_map_log_mean_mins.append(score_log_mean_min.item())

        score_maps_std = score_maps_std.squeeze(0)
        score_maps_mean = score_maps_mean.squeeze(0)
        
        
        log_min = score_log_std_min
        log_max = score_log_std_max
        score_map_std = -torch.log10(score_maps_std)
        spread = (log_max-log_min)
        score_map_std = 255 * (score_map_std-log_min)/spread if spread > 0 else torch.zeros_like(score_map_std)
        
        log_min = score_log_mean_min
        log_max = score_log_mean_max
        score_map_mean = -torch.log10(score_maps_mean)
        spread = (log_max-log_min)
        score_map_mean = 255 * (score_map_mean-log_min)/spread if spread > 0 else torch.zeros_like(score_map_mean)
        print(spread, log_min, log_max)

        for i, render_dir in enumerate(render_dirs):
            if os.path.basename(render_dir) == 'color':
                score_map_dir = os.path.join(os.path.dirname(render_dir), 'score-map')
            else:
                score_map_dir = os.path.join(os.path.dirname(render_dir), 'gt-score-map')
            
            score_map_path = os.path.join(DATA_DIR, score_map_dir, 'stats_' + basename)
            
            log_min = score_log_min[i]
            log_max = score_log_max[i]
            score_map = -torch.log10(score_maps[i])
            spread = (log_max-log_min)
            score_map = 255 * (score_map-log_min)/spread if spread > 0 else torch.zeros_like(score_map)
            
            score_map = torch.concat([score_map_mean, score_map, score_map_std])
            
            score_map = score_map.byte() # quantize
            image = Image.fromarray(score_map.permute([1,2,0]).numpy(), mode='RGB')
            image.save(score_map_path, format='PNG')

    return pd.Series({
        "score_map_log_std_max": score_map_log_std_maxs,
        "score_map_log_std_min": score_map_log_std_mins,
        "score_map_log_mean_max": score_map_log_mean_maxs,
        "score_map_log_mean_min": score_map_log_mean_mins,
    })

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
