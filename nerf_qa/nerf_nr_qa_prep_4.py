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
    score_map_log_mins = group['score_map_log_min'].apply(eval)
    score_map_log_maxs = group['score_map_log_max'].apply(eval)
    for frame in range(frame_count):
        score_maps = []
        for render_dir, score_map_log_min, score_map_log_max in zip(render_dirs, score_map_log_mins, score_map_log_maxs):
            log_min = score_map_log_min[frame].item()
            log_max = score_map_log_max[frame].item()
            if os.path.basename(render_dir) == 'color':
                score_map_dir = os.path.join(os.path.dirname(render_dir), 'score-map')
            else:
                score_map_dir = os.path.join(os.path.dirname(render_dir), 'gt-score-map')
            basename = basenames[frame]
            score_map_path = os.path.join(DATA_DIR, score_map_dir, basename)
            score_map = Image.open(score_map_path)
            score_map = torch.from_numpy(np.array(score_map)).permute(2, 0, 1).float() / 255.0
            assert score_map.shape[0] == 1
            score_map = (log_max-log_min) * score_map + log_min
            score_map = torch.pow(10, score_map)
            score_maps.append(score_map)

        score_maps = np.stack(score_maps)
        score_maps_std = np.std(score_maps, axis=0, keepdims=True)
        score_maps_mean = np.mean(score_maps, axis=0, keepdims=True)
        score_maps = np.concatinate([score_maps, score_maps_std, score_maps_mean], axis=1)

        score_maps_min = score_maps.amin(dim=[2,3]).squeeze(1)
        score_maps_max = score_maps.amax(dim=[2,3]).squeeze(1)
        score_log_max = (-torch.log10(score_maps_min)).numpy().tolist()
        score_log_min = (-torch.log10(score_maps_max)).numpy().tolist()

        score_maps_std_min = score_maps_std.amin(dim=[2,3]).squeeze(1)
        score_maps_std_max = score_maps_std.amax(dim=[2,3]).squeeze(1)
        score_maps_mean_min = score_maps_mean.amin(dim=[2,3]).squeeze(1)
        score_maps_mean_max = score_maps_mean.amax(dim=[2,3]).squeeze(1)
        score_log_std_max = (-torch.log10(score_maps_std_min)).numpy().tolist()
        score_log_std_min = (-torch.log10(score_maps_std_max)).numpy().tolist()
        score_log_mean_max = (-torch.log10(score_maps_mean_min)).numpy().tolist()
        score_log_mean_min = (-torch.log10(score_maps_mean_max)).numpy().tolist()
        
        for i, basename in tqdm(enumerate(basenames)):
            score_map_path = os.path.join(score_map_dir, basename)
            log_min = score_log_min[i]
            log_max = score_log_max[i]
            score_map = -torch.log10(score_maps[i])
            spread = (log_max-log_min)
            score_map = 255 * (score_map-log_min)/spread if spread > 0 else torch.zeros_like(dists_score)
                    
            log_min = score_log_std_min[i]
            log_max = score_log_std_max[i]
            score_map_std = -torch.log10(score_maps_std[i])
            spread = (log_max-log_min)
            score_map_std = 255 * (score_map_std-log_min)/spread if spread > 0 else torch.zeros_like(dists_score)
            
            log_min = score_log_mean_min[i]
            log_max = score_log_mean_max[i]
            score_map_mean = -torch.log10(score_maps_mean[i])
            spread = (log_max-log_min)
            score_map_mean = 255 * (score_map_mean-log_min)/spread if spread > 0 else torch.zeros_like(dists_score)
            
            score_map = torch.concat([score_map_mean, score_map, score_map_std])
            score_map = score_map.byte() # quantize
            score_map = score_map.squeeze([0])
            image = Image.fromarray(score_map.numpy(), mode='RGB')
            image.save(score_map_path, format='PNG')

    return pd.Series({
        "score_map_log_std_max": score_log_std_max.numpy().tolist(),
        "score_map_log_std_min": score_log_std_min.numpy().tolist(),
        "score_map_log_mean_max": score_log_mean_max.numpy().tolist(),
        "score_map_log_mean_min": score_log_mean_min.numpy().tolist(),
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
