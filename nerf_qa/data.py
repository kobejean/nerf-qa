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
from torch.utils.data import Dataset, DataLoader

# data 
import pandas as pd
import cv2
from torch.utils.data import TensorDataset
from tqdm import tqdm
from PIL import Image
import plotly.express as px

from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS, prepare_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%%

class LargeQADataset(Dataset):

    def __init__(self, dir, scores_df, resize=True):
        self.ref_dir = path.join(dir, "references")
        self.dist_dir = path.join(dir, "nerf-renders")
        self.scores_df = scores_df
        self.resize = resize
        self.total_size = self.scores_df['frame_count'].sum()
        self.cumulative_frame_counts = self.scores_df['frame_count'].cumsum()



    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Determine which video the index falls into
        video_idx = (self.cumulative_frame_counts > idx).idxmax()
        if video_idx > 0:
            frame_within_video = idx - self.cumulative_frame_counts.iloc[video_idx - 1]
        else:
            frame_within_video = idx

        # Get the filenames for the distorted and referenced frames
        distorted_filename = self.scores_df.iloc[video_idx]['distorted_filename']
        referenced_filename = self.scores_df.iloc[video_idx]['referenced_filename']

        # Construct the full paths
        distorted_path = os.path.join(self.dist_dir, distorted_filename, f"{frame_within_video:03d}.png")
        referenced_path = os.path.join(self.ref_dir, referenced_filename, f"{frame_within_video:03d}.png")

        # Load and optionally resize images
        distorted_image = prepare_image(Image.open(distorted_path).convert("RGB")).squeeze(0)
        referenced_image = prepare_image(Image.open(referenced_path).convert("RGB")).squeeze(0)

        row = self.scores_df.iloc[video_idx]
        score = row['MOS']
        return distorted_image, referenced_image, score, video_idx
  


#%%
DATA_DIR = "/home/ccl/Datasets/NeRF-QA-Large-1"
REF_DIR = path.join(DATA_DIR, "references")
REND_DIR = path.join(DATA_DIR, "nerf-renders")
SCORE_FILE = path.join(DATA_DIR, "scores.csv")
SCOREs_FILE = path.join(DATA_DIR, "scores_update.csv")
scores_df = pd.read_csv(SCORE_FILE)
# scores_df['scene'] = scores_df['referenced_filename'].str.replace('gt_', '', 1)

# def count_images_in_dir(directory):
#     """Count the number of .png images in the specified directory."""
#     directory = path.join(REF_DIR, directory)
#     return len([name for name in os.listdir(directory) if name.endswith('.png')])

# # Apply the function to each row in the DataFrame to create the 'frame_count' column
# scores_df['frame_count'] = scores_df['referenced_filename'].apply(lambda x: count_images_in_dir(x))

# column_order = ['scene', 'frame_count'] + [col for col in scores_df.columns if not col in ['scene', 'frame_count']]

# scores_df = scores_df[column_order]

#%%
from IPython.display import display
display(scores_df)
print(scores_df.head(3))
print(scores_df['scene'].drop_duplicates().values)
#%%

# %%  
dataset = LargeQADataset(dir = DATA_DIR, scores_df=scores_df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
all_dists = []
all_ids = []
dists_model = DISTS().to(device)
for dist, ref, score, video_idx in iter(dataloader):
    score = dists_model(dist.to(device), ref.to(device))
    all_dists.append(score.cpu().detach().numpy())
    all_ids.append(video_idx.cpu().detach().numpy())

#%%
df = pd.DataFrame({
                'ID': np.concatenate(all_ids, axis=0),
                'DISTS': np.concatenate(all_dists, axis=0),
            })

# Step 2: Group by ID and calculate mean
average_scores = df.groupby('ID').mean().reset_index()
display(average_scores.head(10))
#scores_df.to_csv(SCOREs_FILE, index=False)
# %%
