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
        distorted_image = prepare_image(Image.open(distorted_path).convert("RGB"), resize=self.resize).squeeze(0)
        referenced_image = prepare_image(Image.open(referenced_path).convert("RGB"), resize=self.resize).squeeze(0)

        row = self.scores_df.iloc[video_idx]
        score = row['MOS']
        return distorted_image, referenced_image, score, video_idx


class ComputeBatchSampler(Sampler):
    def __init__(self, dataset, compute_batch_size):
        self.dataset = dataset
        self.compute_batch_size = compute_batch_size

        # Organize indices by image size (assuming dataset[idx] returns a tuple (image, label))
        self.indices_by_size = {}
        for idx in tqdm(range(len(dataset)), desc="Preparing Sampler..."):
            image = dataset[idx][0]
            size = tuple(image.size())
            if size not in self.indices_by_size:
                self.indices_by_size[size] = []
            self.indices_by_size[size].append(idx)

        self.batches = self._create_batches()

    def _create_batches(self):
        # This method should organize indices into larger batches ensuring diversity in dimensions
        # and grouping them into mini-batches by size for computational efficiency
        batches = []
        # Example logic (simplified and needs to be optimized):
        for size, indices in self.indices_by_size.items():
            for i in range(0, len(indices), self.compute_batch_size):
                batches.append(indices[i:i + self.compute_batch_size])
        return batches
    
    def __iter__(self):
        np.random.shuffle(self.batches)  # Shuffle to ensure diversity in each larger batch
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


# Batch creation function
def create_large_qa_dataloader(scores_df, dir, resize=True):
    # Create a dataset and dataloader for efficient batching
    dataset = LargeQADataset(dir=dir, scores_df=scores_df, resize=resize)
    sampler = ComputeBatchSampler(dataset, DEVICE_BATCH_SIZE)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
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
def create_test_video_dataloader(row, dir, resize=True):
    ref_dir = path.join(dir, "Reference")
    syn_dir = path.join(dir, "NeRF-QA_videos")
    dist_video_path = path.join(syn_dir, row['distorted_filename'])
    ref_video_path = path.join(ref_dir, row['reference_filename'])
    ref = load_video_frames(ref_video_path, resize=resize)
    dist = load_video_frames(dist_video_path, resize=resize)
    # Create a dataset and dataloader for efficient batching
    dataset = TensorDataset(dist, ref)
    dataloader = DataLoader(dataset, batch_size=DEVICE_BATCH_SIZE, shuffle=False)
    return dataloader
#%%
if __name__ == "__main__":

    DATA_DIR = "/home/ccl/Datasets/NeRF-QA-Large-1"
    SCORE_FILE = path.join(DATA_DIR, "scores.csv")
    # Read the CSV file
    scores_df = pd.read_csv(SCORE_FILE)
    # filter test
    test_scenes = ['ship', 'lego', 'drums', 'ficus', 'train', 'm60', 'playground', 'truck']
    train_df = scores_df[~scores_df['scene'].isin(test_scenes)].reset_index() # + ['trex', 'horns']
    # val_df = scores_df[scores_df['scene'].isin(test_scenes)].reset_index()
    dataloader = create_large_qa_dataloader(train_df, dir=DATA_DIR)
    dist, ref, score, id = next(iter(dataloader))
    print(dist, ref, score, id)



# %%
