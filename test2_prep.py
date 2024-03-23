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

        gt_files = [f for f in os.listdir(gt_dir) if f.endswith((".jpg", ".png"))]
        gt_files.sort()
        render_files = [f for f in os.listdir(render_dir) if f.endswith((".jpg", ".png"))]
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
    syn_dir = path.join(dir, "NeRF-QA_videos")
    dist_path = path.join(syn_dir, row['distorted_filename'])
    ref_path = path.join(ref_dir, row['reference_filename'])
     
    # Create a dataset and dataloader for efficient batching
    dataset = CustomDataset(ref_path, dist_path)
    dataloader = DataLoader(dataset, batch_size=DEVICE_BATCH_SIZE, shuffle=False, collate_fn = recursive_collate)
    return dataloader  

TEST_DATA_DIR = "/home/ccl/Datasets/NeRF-QA"
TEST_SCORE_FILE = path.join(TEST_DATA_DIR, "NeRF_VQA_MOS.csv")
test_df = pd.read_csv(TEST_SCORE_FILE)
test_size = test_df.shape[0]
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
