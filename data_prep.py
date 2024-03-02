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

from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS, prepare_image
from nerf_qa.settings import DEVICE_BATCH_SIZE
from nerf_qa.data import LargeQADataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "/home/ccl/Datasets/NeRF-QA-Large-1"
SCORE_FILE = path.join(DATA_DIR, "scores.csv")
# Read the CSV file
scores_df = pd.read_csv(SCORE_FILE)

# Initialize the model
dists_model = DISTS().to(device)

# Initialize an empty list to store the scores
dists_scores = []

# Iterate over each row in the DataFrame
for index, row in scores_df.iterrows():
    ref_path = path.join(DATA_DIR, 'references', row['referenced_filename'])
    dist_path = path.join(DATA_DIR, 'nerf-renders', row['distorted_filename'])
    frame_count = row['frame_count']
    scores = []
    
    # Process each frame
    for i in range(frame_count):
        ref_file = path.join(ref_path, f"{i:03d}.png")
        dist_file = path.join(dist_path, f"{i:03d}.png")
        
        # Load and optionally resize images
        distorted_image = prepare_image(Image.open(dist_file).convert("RGB"), resize=False)
        referenced_image = prepare_image(Image.open(ref_file).convert("RGB"), resize=False)
        
        # Calculate the score
        score = dists_model(distorted_image.to(device), referenced_image.to(device), require_grad=False, batch_average=False)
        scores.append(score.cpu().item())
    
    # Calculate the mean score for the current row and append it to the list
    dists_scores.append(np.mean(scores))

# Assign the list of calculated mean scores to a new column in the DataFrame
scores_df['DISTS_no_resize'] = dists_scores


# %%
#scores_df["DISTS_old"] = scores_df['DISTS']
#scores_df["DISTS"] = scores_df['DISTS_no_resize']

# %%
scores_df.to_csv(path.join(DATA_DIR, "scores_new.csv"))
# %%
scores_df.head(10)

# %%
