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

def compute_correlations(pred_scores, mos):
    plcc = pearsonr(pred_scores, mos)[0]
    srcc = spearmanr(pred_scores, mos)[0]
    ktcc = kendalltau(pred_scores, mos)[0]

    return {
        'plcc': plcc,
        'srcc': srcc,
        'ktcc': ktcc,
    }
# Example function to load a video and process it frame by frame
def load_video_frames(video_path, resize=True, keep_aspect_ratio=False):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to RGB (from BGR) and then to tensor
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frame = transforms.ToPILImage()(frame)
        frame = transforms.ToTensor()(frame)
        if keep_aspect_ratio:
            render_256 = F.interpolate(frame.unsqueeze(0), size=(256), mode='bilinear', align_corners=False).squeeze(0)
        else:
            render_256 = F.interpolate(frame.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        #render_224 = F.interpolate(frame.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        #render = { "256x256": render_256, "224x224": render_224 }
        frames.append(render_256)
    cap.release()
    return frames

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Retrieve the data row at the given index
        data_row = self.data[index]
        return data_row
    
from PIL import Image

def load_image(path):
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
    image = F.interpolate(image.unsqueeze(0), size=(256), mode='bilinear', align_corners=False).squeeze(0)

    return image

test_video_dataloader_tr_map = {
    'mic': '/home/ccl/Datasets/NeRF/blender_test/mic/test',
    'hotdog': '/home/ccl/Datasets/NeRF/blender_test/hotdog/test',
    'materials': '/home/ccl/Datasets/NeRF/blender_test/materials/test',
    'drums': '/home/ccl/Datasets/NeRF/blender_test/drums/test',
    'ship': '/home/ccl/Datasets/NeRF/blender_test/ship/test',
    'lego': '/home/ccl/Datasets/NeRF/blender_test/lego/test',
    'chair': '/home/ccl/Datasets/NeRF/blender_test/chair/test',
}

# Batch creation function
def create_test_video_dataloader(row, dir, resize=True, keep_aspect_ratio=False):
    ref_dir = path.join(dir, "Reference")
    syn_dir = path.join(dir, "NeRF-QA_videos")
    dist_video_path = path.join(syn_dir, row['distorted_filename'])
    dist = load_video_frames(dist_video_path, resize=resize, keep_aspect_ratio=keep_aspect_ratio)
    if False and row['scene'] in test_video_dataloader_tr_map:
        gt_dir = test_video_dataloader_tr_map[row['scene']]
        gt_files = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith((".jpg", ".png"))]
        gt_files.sort()
        ref = [load_image(gt_file) for gt_file in gt_files]
    else:
        ref_video_path = path.join(ref_dir, row['reference_filename'])
        ref = load_video_frames(ref_video_path, resize=resize, keep_aspect_ratio=keep_aspect_ratio)
    # Create a dataset and dataloader for efficient batching
    dataset = CustomDataset(list(zip(ref, dist)))
    dataloader = DataLoader(dataset, batch_size=DEVICE_BATCH_SIZE, shuffle=False, collate_fn = recursive_collate)
    return dataloader  

TEST_DATA_DIR = "/home/ccl/Datasets/NeRF-QA"
TEST_SCORE_FILE = path.join(TEST_DATA_DIR, "NeRF_VQA_MOS.csv")
test_df = pd.read_csv(TEST_SCORE_FILE)
test_df['scene'] = test_df['reference_filename'].str.replace('_reference.mp4', '', regex=False)
test_size = test_df.shape[0]
adists_model = ADISTS().to(device)
dists_model = DISTS().to(device)
video_adists_scores = []
video_dists_scores = []
video_adists_scores_std = []
video_dists_scores_std = []
video_adists_scores_max = []
video_dists_scores_max = []
video_adists_scores_min = []
video_dists_scores_min = []
for index, row in tqdm(test_df.iterrows(), total=test_size, desc="Processing..."):
    frames_data = create_test_video_dataloader(row, TEST_DATA_DIR)
    frame_adists_scores = []
    frame_dists_scores = []
    for ref, render in frames_data:
        batch_adists_scores = adists_model(ref.to(device), render.to(device), as_loss=False)
        frame_adists_scores.append(batch_adists_scores.detach().cpu().numpy())

        batch_dists_scores = dists_model(ref.to(device), render.to(device), batch_average=False)
        frame_dists_scores.append(batch_dists_scores.detach().cpu().numpy())
    video_adists_score = np.mean(np.concatenate(frame_adists_scores))
    video_dists_score = np.mean(np.concatenate(frame_dists_scores))
    video_adists_score_std = np.std(np.concatenate(frame_adists_scores))
    video_dists_score_std = np.std(np.concatenate(frame_dists_scores))
    video_adists_score_min = np.min(np.concatenate(frame_adists_scores))
    video_dists_score_min = np.min(np.concatenate(frame_dists_scores))
    video_adists_score_max = np.max(np.concatenate(frame_adists_scores))
    video_dists_score_max = np.max(np.concatenate(frame_dists_scores))
    print(video_adists_score, batch_adists_scores)
    print(video_dists_score, batch_dists_scores)
    video_adists_scores.append(video_adists_score)
    video_dists_scores.append(video_dists_score)
    video_adists_scores_std.append(video_adists_score_std)
    video_dists_scores_std.append(video_dists_score_std)
    video_adists_scores_min.append(video_adists_score_min)
    video_dists_scores_min.append(video_dists_score_min)
    video_adists_scores_max.append(video_adists_score_max)
    video_dists_scores_max.append(video_dists_score_max)
test_df['A-DISTS'] = video_adists_scores
test_df['DISTS'] = video_dists_scores
test_df['A-DISTS_std'] = video_adists_scores_std
test_df['DISTS_std'] = video_dists_scores_std
test_df['A-DISTS_min'] = video_adists_scores_min
test_df['DISTS_min'] = video_dists_scores_min
test_df['A-DISTS_max'] = video_adists_scores_max
test_df['DISTS_max'] = video_dists_scores_max

#%%
TEST_DATA_DIR = "/home/ccl/Datasets/NeRF-QA"
TEST_SCORE_FILE = path.join(TEST_DATA_DIR, "NeRF_VQA_MOS.csv")
test_df = pd.read_csv(TEST_SCORE_FILE)
test_df['scene'] = test_df['reference_filename'].str.replace('_reference.mp4', '', regex=False)
test_size = test_df.shape[0]
#%%
syn_files = ['ficus_reference.mp4', 'ship_reference.mp4',
 'drums_reference.mp4']
tnt_files = ['truck_reference.mp4', 'playground_reference.mp4',
 'train_reference.mp4', 'm60_reference.mp4']
print(test_df['reference_filename'].unique())

syn_df = test_df[test_df['reference_filename'].isin(syn_files)].reset_index()
tnt_df = test_df[test_df['reference_filename'].isin(tnt_files)].reset_index()
#%%
display(test_df['reference_filename'].isin(syn_files))
#%%
corr = compute_correlations(syn_df['A-DISTS'], syn_df['MOS'])
print("syn a-dists mos", corr)
corr = compute_correlations(tnt_df['A-DISTS'], tnt_df['MOS'])
print("tnt a-dists mos", corr)
corr = compute_correlations(test_df['A-DISTS'], test_df['MOS'])
print("all a-dists mos", corr)
corr = compute_correlations(syn_df['A-DISTS'], syn_df['DMOS'])
print("syn a-dists dmos", corr)
corr = compute_correlations(tnt_df['A-DISTS'], tnt_df['DMOS'])
print("tnt a-dists dmos", corr)
corr = compute_correlations(test_df['A-DISTS'], test_df['DMOS'])
print("all a-dists dmos", corr)
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
metrics = ['FVVHD', 'Contrique_score', 'WaDiQaM_score', 'SSIM', 'GMSD', 'MS-SSIM_Score', 'PSNR_Score', 'LPIPS_Score', 'DISTS', 'LPIPS_Score_vgg', 'WaDiQa_score', 'CompressVQA']
data = []

# Assuming syn_df, tnt_df, and test_df are your DataFrames with the data
for metric in metrics:
    correlations = get_correlations(metric, syn_df, tnt_df, test_df)
    correlations['Metric'] = metric
    data.append(correlations)

# Creating the DataFrame
df_corr = pd.DataFrame(data)
df_corr = df_corr.set_index('Metric')
df_corr
# %%
df_corr.to_csv('correlations.csv')

# %%
