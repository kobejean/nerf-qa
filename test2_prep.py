#%%
# system level
import os
from os import path
import sys
import argparse


# deep learning
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import wandb
from sklearn.linear_model import LinearRegression

# data 
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler

import argparse
import wandb
from torch.profiler import profile, record_function, ProfilerActivity
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# local
from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS
from nerf_qa.ADISTS import ADISTS
from nerf_qa.data import NerfNRQADataset, SceneBalancedSampler
from nerf_qa.logger import MetricCollectionLogger
from nerf_qa.settings import DEVICE_BATCH_SIZE
from nerf_qa.model_nr_v6 import NRModel
import multiprocessing as mp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def recursive_collate(batch):
    if isinstance(batch[0], torch.Tensor):
        return (torch.stack(batch) if batch[0].dim() == 0 or batch[0].shape[0] > 1 else torch.concat(batch, dim=0)).detach()
    elif isinstance(batch[0], tuple):
        return tuple(recursive_collate(samples) for samples in zip(*batch))
    elif isinstance(batch[0], list):
        return [recursive_collate(samples) for samples in zip(*batch)]
    elif isinstance(batch[0], dict):
        return {key: recursive_collate([sample[key] for sample in batch]) for key in batch[0]}
    else:
        return batch
    
def batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (tuple, list)):
        # Handle tuples and lists by recursively calling batch_to_device on each element
        # Only recurse if the elements are not tensors
        processed = [batch_to_device(item, device) if not isinstance(item, torch.Tensor) else item.to(device) for item in batch]
        return type(batch)(processed)
    elif isinstance(batch, dict):
        # Handle dictionaries by recursively calling batch_to_device on each value
        return {key: batch_to_device(value, device) for key, value in batch.items()}
    else:
        # Return the item unchanged if it's not a tensor, list, tuple, or dict
        return batch
    
from torch.utils.data import Dataset, TensorDataset, DataLoader, Sampler
import cv2
from torchvision import models,transforms
import torch.nn.functional as F
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

class CustomDataset(Dataset):
    def __init__(self, gt_dir, render_dir):

        gt_files = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith((".jpg", ".png"))]
        gt_files.sort()
        render_files = [os.path.join(render_dir, f) for f in os.listdir(render_dir) if f.endswith((".jpg", ".png"))]
        render_files.sort()
        frame_count = max(len(gt_files), len(render_files))

        self.files = list(zip(gt_files, render_files))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        # Retrieve the data row at the given index
        gt_path, render_path = self.files[index]
        gt = self.load_image(gt_path)
        render = self.load_image(render_path)
        return gt, render
    
    def load_image(self, path):
        image = Image.open(path)

        if image.mode == 'RGBA':
            # If the image has an alpha channel, create a white background
            background = Image.new('RGBA', image.size, (255, 255, 255))
            
            # Paste the image onto the white background using alpha compositing
            background.paste(image, mask=image.split()[3])
            
            # Convert the image to RGB mode
            image = background.convert('RGB')
        else:
            # If the image doesn't have an alpha channel, directly convert it to RGB
            image = image.convert('RGB')

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image = F.interpolate(image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)

        return image

# Batch creation function
def create_test_dataloader(row, dir):
    ref_dir = path.join(dir, "Reference")
    syn_dir = path.join(dir, "Renders")
    dist_path = path.join(syn_dir, row['distorted_folder'])
    ref_path = path.join(ref_dir, row['reference_folder'])
     
    # Create a dataset and dataloader for efficient batching
    dataset = CustomDataset(ref_path, dist_path)
    dataloader = DataLoader(dataset, batch_size=DEVICE_BATCH_SIZE, shuffle=False, collate_fn = recursive_collate)
    return dataloader  

TEST_DATA_DIR = "/home/ccl/Datasets/Test_2-datasets/"
TEST_SCORE_FILE = path.join(TEST_DATA_DIR, "scores.csv")
test_df = pd.read_csv(TEST_SCORE_FILE)
test_size = test_df.shape[0]
#%%
adists_model = ADISTS().to(device)
dists_model = DISTS().to(device)
video_adists_scores = []
video_dists_scores = []
for index, row in tqdm(test_df.iterrows(), total=test_size, desc="Processing..."):
    frames_data = create_test_dataloader(row, TEST_DATA_DIR)
    frame_adists_scores = []
    frame_dists_scores = []
    for ref, render in frames_data:
        batch_adists_scores = adists_model(ref.to(device), render.to(device), as_loss=False)
        frame_adists_scores.append(batch_adists_scores.detach().cpu().numpy())

        batch_dists_scores = dists_model(ref.to(device), render.to(device), batch_average=False)
        frame_dists_scores.append(batch_dists_scores.detach().cpu().numpy())
    video_adists_score = np.mean(np.concatenate(frame_adists_scores))
    video_dists_score = np.mean(np.concatenate(frame_dists_scores))
    print(video_adists_score, batch_adists_scores)
    print(video_dists_score, batch_dists_scores)
    video_adists_scores.append(video_adists_score)
    video_dists_scores.append(video_dists_score)
test_df['A-DISTS'] = video_adists_scores
test_df['DISTS'] = video_dists_scores

#%%
display(test_df.head(3))
#%%
syn_files = ['gt_chair', 'gt_mic', 'gt_hotdog', 'gt_materials']
tnt_files = ['gt_horns', 'gt_trex', 'gt_fortress', 'gt_room']
print(test_df['reference_folder'].unique())
#%%
syn_df = test_df[test_df['reference_folder'].isin(syn_files)].reset_index()
tnt_df = test_df[test_df['reference_folder'].isin(tnt_files)].reset_index()
#%%
display(test_df['reference_folder'].isin(syn_files))
#%%
corr = compute_correlations(syn_df['A-DISTS'], syn_df['MOS'])
print("syn a-dists mos", corr)
corr = compute_correlations(tnt_df['A-DISTS'], tnt_df['MOS'])
print("tnt a-dists mos", corr)
corr = compute_correlations(test_df['A-DISTS'], test_df['MOS'])
print("all a-dists mos", corr)

#%%
corr = compute_correlations(syn_df['DISTS'], syn_df['MOS'])
print("syn dists mos", corr)
corr = compute_correlations(tnt_df['DISTS'], tnt_df['MOS'])
print("tnt dists mos", corr)
corr = compute_correlations(test_df['DISTS'], test_df['MOS'])
print("all dists mos", corr)
# %%
#%%
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go

def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha])

def plot_dists_mos_with_group_regression_b_ave(df, y_col='DISTS', x_col='MOS', group_col='reference_folder'):
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

    fig = go.Figure()

    unique_groups = df[group_col].unique()
    for i, group in enumerate(unique_groups):
        group_df = df[df[group_col] == group]
        group_x = group_df[x_col]
        group_y = group_df[y_col]
        
        params, params_covariance = curve_fit(linear_func, group_x, group_y)
        
        x_range = np.linspace(min(group_x), max(group_x), 400)
        y_pred = linear_func(x_range, *params)
        
        color = colors[i % len(colors)]
        rgba_color = hex_to_rgba(color, 0.5)  # Adjust the alpha value to 0.5 for transparency
        
        fig.add_trace(go.Scatter(x=group_x, y=group_y, mode='markers', name=f'Data: {group}', marker_color=color))
        fig.add_trace(go.Scatter(x=group_x, y=group_y, mode='lines', line=dict(color=rgba_color, width=2)))
        
        fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name=f'Regression: {group}', line=dict(color=color)))

    fig.update_layout(title=f'Linear Regression per Group between {y_col} and {x_col}',
                      xaxis_title=x_col,
                      yaxis_title=y_col)
    return fig

display(plot_dists_mos_with_group_regression_b_ave(test_df, 'DISTS', 'MOS'))
display(plot_dists_mos_with_group_regression_b_ave(test_df, 'A-DISTS', 'MOS'))

# %%
