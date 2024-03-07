#%%
# system level
import os
from os import path
import sys
import argparse


# deep learning
from scipy.stats import pearsonr, spearmanr
import numpy as np
import torch
from torch import nn
from torchvision import models,transforms
import torch.optim as optim
import wandb
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset, DataLoader, Sampler

# data 
import pandas as pd
import cv2
from torch.utils.data import TensorDataset
from tqdm import tqdm
from PIL import Image
import plotly.express as px
import re

from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS, prepare_image
from nerf_qa.settings import DEVICE_BATCH_SIZE
from nerf_qa.data import LargeQADataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "/home/ccl/Datasets/NeRF-QA-Large-1"
SCORE_FILE = path.join(DATA_DIR, "scores.csv")
# Read the CSV file
scores_df = pd.read_csv(SCORE_FILE)

frame_files_patt = re.compile(r'^\d{3}\.png$')
def frame_filepaths(dir):
    matches = [f for f in os.listdir(dir) if frame_files_patt.match(f)]
    return [path.join(dir, match) for match in matches]

# Initialize the model
dists_model = DISTS().to(device)
#%%
# Initialize an empty list to store the scores
dists_scores = []
dists_scores_no_resize = []
frame_counts = []
frame_heights = []
frame_widths = []
scenes = []

# Iterate over each row in the DataFrame
for index, row in tqdm(scores_df.iterrows(), total=len(scores_df), position=0):
    ref_filename = row['referenced_filename']
    dist_filename = row['distorted_filename']
    print(dist_filename)
    ref_path = path.join(DATA_DIR, 'references', ref_filename)
    dist_path = path.join(DATA_DIR, 'nerf-renders', dist_filename)
    scores_no_resize = []
    scores = []
    ref_paths = frame_filepaths(ref_path)
    dist_paths = frame_filepaths(dist_path)
    frame_count = len(ref_paths)
    scene = ref_filename[3:] if ref_filename.startswith('gt_') else ref_filename
    
    # Process each frame
    for ref_file, dist_file in tqdm(zip(ref_paths, dist_paths), position=0, total=frame_count, leave=False):        
        
        # Load and optionally resize images
        distorted_image = prepare_image(Image.open(dist_file).convert("RGB"), resize=True)
        referenced_image = prepare_image(Image.open(ref_file).convert("RGB"), resize=True)
        
        # Calculate the score
        score = dists_model(distorted_image.to(device), referenced_image.to(device), require_grad=False, batch_average=False)
        scores.append(score.cpu().item())

        # Load and optionally resize images
        distorted_image = prepare_image(Image.open(dist_file).convert("RGB"), resize=False)
        referenced_image = prepare_image(Image.open(ref_file).convert("RGB"), resize=False)
        
        # Calculate the score
        score = dists_model(distorted_image.to(device), referenced_image.to(device), require_grad=False, batch_average=False)
        scores_no_resize.append(score.cpu().item())
    #print(referenced_image.shape)

    # Calculate the mean score for the current row and append it to the list
    dists_scores.append(np.mean(scores))
    dists_scores_no_resize.append(np.mean(scores_no_resize))
    frame_counts.append(frame_count)
    frame_heights.append(referenced_image.shape[-2])
    frame_widths.append(referenced_image.shape[-1])
    scenes.append(scene)

# Assign the list of calculated mean scores to a new column in the DataFrame
scores_df['DISTS_new'] = dists_scores
scores_df['DISTS_no_resize'] = dists_scores_no_resize
scores_df['frame_height'] = frame_heights
scores_df['frame_width'] = frame_widths
scores_df['frame_count'] = frame_counts
scores_df['scene'] = scenes


# %%

# %%
order = [
    'distorted_filename', 'referenced_filename', 'scene',
    'frame_count', 'frame_height', 'frame_width', 'MOS',
    'DISTS', 'DISTS_new', 'DISTS_no_resize'
         ]
columns = order + [col for col in scores_df.columns if col not in order]
scores_df = scores_df[columns]
scores_df
# %%
# %%
scores_df.to_csv(path.join(DATA_DIR, "scores_new.csv"))

# %%
