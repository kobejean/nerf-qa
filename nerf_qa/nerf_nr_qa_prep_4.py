#%%
import os
import csv
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF

from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS, prepare_image
from nerf_qa.ADISTS import ADISTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
adists_model = ADISTS().to(device)
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


def load_image(path):
    image = Image.open(path)

    if image.mode == 'RGBA':
        background = Image.new('RGBA', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background.convert('RGB')
    else:
        image = image.convert('RGB')
    return image

def compute_std_mean(group):
    frame_count = group['frame_count'].values[0]
    basenames = eval(group['basenames'].values[0])
    render_dirs = group['render_dir'].values
    gt_dirs = group['gt_dir'].values
    score_map_log_mins = [[] for _ in range(len(render_dirs))] 
    score_map_log_maxs = [[] for _ in range(len(render_dirs))] 
    score_map_log_std_maxs = []
    score_map_log_std_mins = []
    score_map_log_mean_maxs = []
    score_map_log_mean_mins = []

    print(group['scene'].unique())
    for frame in tqdm(range(frame_count)):
        score_maps = []
        for gt_dir, render_dir in zip(gt_dirs, render_dirs):
            basename = basenames[frame]
            gt_im = prepare_image(load_image(os.path.join(DATA_DIR, gt_dir, basename)), resize=False)
            render_im = prepare_image(load_image(os.path.join(DATA_DIR, render_dir, basename)), resize=False)
            
            h, w = (int(render_im.shape[2]*0.7), int(render_im.shape[3]*0.7))
            i, j = (render_im.shape[2]-h)//2, (render_im.shape[3]-w)//2
            # Crop to avoid black region due to postprocessed distortion
            render_im = TF.crop(render_im, i, j, h, w)
            gt_im = TF.crop(gt_im, i, j, h, w)
            #render_im = TF.resize(render_im,(256, 256))
            #gt_im = TF.resize(gt_im,(256, 256))
            with torch.no_grad():
                score_map = adists_model(render_im.to(device), gt_im.to(device), as_map=True)
            score_maps.append(score_map.clamp(1e-10).detach().cpu().double())

        score_maps = torch.concat(score_maps, dim=0)
        score_map_std = torch.std(score_maps, dim=0, keepdim=True)
        score_map_mean = torch.mean(score_maps, dim=0, keepdim=True)
        #score_maps = torch.concat([score_maps, score_map_std, score_map_mean], dim=1)
        
        score_maps_min = score_maps.amin(dim=[2,3]).squeeze(1)
        score_maps_max = score_maps.amax(dim=[2,3]).squeeze(1)
        score_log_max = (-torch.log10(score_maps_min)).numpy().tolist()
        score_log_min = (-torch.log10(score_maps_max)).numpy().tolist()

        for i, (log_max, log_min) in enumerate(zip(score_log_max, score_log_min)):
            score_map_log_maxs[i].append(log_max)
            score_map_log_mins[i].append(log_min)

        score_map_std_min = score_map_std.amin(dim=[2,3]).squeeze()
        score_map_std_max = score_map_std.amax(dim=[2,3]).squeeze()
        score_map_mean_min = score_map_mean.amin(dim=[2,3]).squeeze()
        score_map_mean_max = score_map_mean.amax(dim=[2,3]).squeeze()
        score_log_std_max = (-torch.log10(score_map_std_min))
        score_log_std_min = (-torch.log10(score_map_std_max))
        score_log_mean_max = (-torch.log10(score_map_mean_min))
        score_log_mean_min = (-torch.log10(score_map_mean_max))

        score_map_log_std_maxs.append(score_log_std_max.item())
        score_map_log_std_mins.append(score_log_std_min.item())
        score_map_log_mean_maxs.append(score_log_mean_max.item())
        score_map_log_mean_mins.append(score_log_mean_min.item())

        score_map_std = score_map_std.squeeze(0)
        score_map_mean = score_map_mean.squeeze(0)
        
        
        log_min = score_log_std_min
        log_max = score_log_std_max
        score_map_std = -torch.log10(score_map_std)
        spread = (log_max-log_min)
        score_map_std = 255 * (log_max - score_map_std)/spread if spread > 0 else torch.zeros_like(score_map_std)
        
        log_min = score_log_mean_min
        log_max = score_log_mean_max
        score_map_mean = -torch.log10(score_map_mean)
        spread = (log_max-log_min)
        score_map_mean = 255 * (log_max - score_map_mean)/spread if spread > 0 else torch.zeros_like(score_map_mean)

        for i, render_dir in enumerate(render_dirs):
            if os.path.basename(render_dir) == 'color':
                score_map_dir = os.path.join(os.path.dirname(render_dir), 'score-map')
            else:
                score_map_dir = os.path.join(os.path.dirname(render_dir), 'gt-score-map')
            
            score_map_path = os.path.join(DATA_DIR, score_map_dir, basename)
            
            log_min = score_log_min[i]
            log_max = score_log_max[i]
            score_map = -torch.log10(score_maps[i])
            spread = (log_max-log_min)
            score_map = 255 * (log_max - score_map)/spread if spread > 0 else torch.zeros_like(score_map)
            
            score_map = torch.concat([score_map_mean, score_map, score_map_std])
            
            score_map = score_map.byte() # quantize
            image = Image.fromarray(score_map.permute([1,2,0]).numpy(), mode='RGB')
            image.save(score_map_path, format='PNG')

    def to_str(array):
        array = ['{:.4e}'.format(num) for num in array]
        return str(array)
    
    group["score_map_log_max"] = list(map(to_str, score_map_log_maxs))
    group["score_map_log_min"] = list(map(to_str, score_map_log_mins))
    group["score_map_log_std_max"] = to_str(score_map_log_std_maxs)
    group["score_map_log_std_min"] = to_str(score_map_log_std_mins)
    group["score_map_log_mean_max"] = to_str(score_map_log_mean_maxs)
    group["score_map_log_mean_min"] = to_str(score_map_log_mean_mins)
    print(len(group))
    return group

black_list_train_methods = [
        #'instant-ngp-10', 'instant-ngp-20', 'instant-ngp-50', 'instant-ngp-100', 'instant-ngp-200', 
        #'nerfacto-10', 'nerfacto-20', 'nerfacto-50', 'nerfacto-100', 'nerfacto-200', 
    ]
filtered_df = df[~df['method'].isin(black_list_train_methods)].reset_index()

# Group by 'scene', apply the compute_std_mean function, and reset index to flatten the DataFrame
result = filtered_df.groupby('scene').apply(compute_std_mean)
result_ = result.reset_index(drop=True)
#%%
display(result_.head(2))
#%%
result_.to_csv("/home/ccl/Datasets/NeRF-NR-QA/output_ADISTS.csv")

#%%
df = df.drop(['DISTS_std', 'DISTS_mean'], axis=1)
df_result = df.merge(result, on='scene')
display(df_result)
#%%

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
