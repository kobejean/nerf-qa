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
from nerf_qa.settings import DEVICE_BATCH_SIZE

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
  
# Batch creation function
def create_large_qa_dataloader(scores_df, dir):
    # Create a dataset and dataloader for efficient batching
    dataset = LargeQADataset(dir=dir, scores_df=scores_df)
    dataloader = DataLoader(dataset, batch_size=DEVICE_BATCH_SIZE, shuffle=True)
    return dataloader

# Example function to load a video and process it frame by frame
def load_video_frames(video_path, resize=True):
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
        frame = prepare_image(frame, resize=resize).squeeze(0)
        frames.append(frame)
    cap.release()
    return torch.stack(frames)


# Batch creation function
def create_test_video_dataloader(row, dir):
    ref_dir = path.join(dir, "Reference")
    syn_dir = path.join(dir, "NeRF-QA_videos")
    dist_video_path = path.join(syn_dir, row['distorted_filename'])
    ref_video_path = path.join(ref_dir, row['reference_filename'])
    ref = load_video_frames(ref_video_path)
    dist = load_video_frames(dist_video_path)
    # Create a dataset and dataloader for efficient batching
    dataset = TensorDataset(dist, ref)
    dataloader = DataLoader(dataset, batch_size=DEVICE_BATCH_SIZE, shuffle=False)
    return dataloader