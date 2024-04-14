
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
import torch.nn.functional as F
from torch import nn
from torchvision import models,transforms
import torch.optim as optim
import wandb
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision.transforms.functional as TF
import random

# data 
import pandas as pd
import cv2
from torch.utils.data import TensorDataset
from tqdm import tqdm
from PIL import Image
import plotly.express as px

from nerf_qa.DISTS_pytorch.DISTS_pt_original import prepare_image
from nerf_qa.settings import DEVICE_BATCH_SIZE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%%
class Test2Dataset(Dataset):
    def __init__(self, row, dir):
        gt_dir = path.join(dir, "Reference", row['reference_folder'])
        render_dir = path.join(dir, "Renders", row['distorted_folder'])

        gt_files = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith((".jpg", ".png"))]
        gt_files.sort()
        render_files = [os.path.join(render_dir, f) for f in os.listdir(render_dir) if f.endswith((".jpg", ".png"))]
        render_files.sort()

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
        image = Image.open(path).convert("RGB")
        image = prepare_image(image, resize=True)
        return image
    
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
    
# Batch creation function
def create_test2_dataloader(row, dir, batch_size):
    # Create a dataset and dataloader for efficient batching
    dataset = Test2Dataset(row, dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn = recursive_collate)
    return dataloader 

class NeRFQAResizedDataset(Dataset):

    def __init__(self, dir, scores_df):
        self.ref_dir = path.join(dir, "Reference")
        self.dist_dir = path.join(dir, "NeRF-QA_videos")
        self.scores_df = scores_df
        
        def get_files(row, base_dir, column_name):
            folder_path = os.path.join(base_dir, row[column_name], '256x256')
            file_list = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png"))]
            file_list.sort()
            return file_list
        
        self.scores_df['distorted_folder'] = self.scores_df['distorted_filename'].apply(lambda x: os.path.splitext(x)[0])
        self.scores_df['reference_folder'] = self.scores_df['reference_filename'].apply(lambda x: os.path.splitext(x)[0])

        self.scores_df['render_files'] = self.scores_df.apply(get_files, axis=1, args=(self.dist_dir, 'distorted_folder'))
        self.scores_df['gt_files'] = self.scores_df.apply(get_files, axis=1, args=(self.ref_dir, 'reference_folder'))
        self.scores_df['frame_count'] = self.scores_df['gt_files'].apply(len)
        self.total_size = self.scores_df['frame_count'].sum()
        self.cumulative_frame_counts = self.scores_df['frame_count'].cumsum()
        self.static_transforms = transforms.Compose([
            # transforms.Resize(256), 
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.total_size
    

    def transform_pair(self, render_image, reference_image):
        # C,H,W = render_image.shape
        # min_length = min(H,W)
        # resize_length = random.randint(256, min_length)
        # render_image = TF.resize(render_image, resize_length)
        # reference_image = TF.resize(reference_image, resize_length)
        i, j, h, w = transforms.RandomCrop.get_params(render_image, output_size=(256,256))
        assert h == 256 and w == 256
        render_image = TF.crop(render_image, i, j, h, w)
        reference_image = TF.crop(reference_image, i, j, h, w)
        return render_image, reference_image

    def __getitem__(self, idx):
        # Determine which video the index falls into
        video_idx = (self.cumulative_frame_counts > idx).idxmax()
        if video_idx > 0:
            frame_within_video = idx - self.cumulative_frame_counts.iloc[video_idx - 1]
        else:
            frame_within_video = idx

        # Get the filenames for the distorted and referenced frames
        distorted_foldername = self.scores_df.iloc[video_idx]['distorted_folder']
        referenced_foldername = self.scores_df.iloc[video_idx]['reference_folder']
        distorted_filename = f'{frame_within_video:03d}.png'
        referenced_filename = f'{frame_within_video:03d}.png'

        # Construct the full paths
        distorted_path = os.path.join(self.dist_dir, distorted_foldername, '256x256', distorted_filename)
        referenced_path = os.path.join(self.ref_dir, referenced_foldername, '256x256', referenced_filename)

        # Load and optionally resize images
        distorted_image = self.static_transforms(Image.open(distorted_path).convert("RGB"))
        referenced_image = self.static_transforms(Image.open(referenced_path).convert("RGB"))
        distorted_image, referenced_image = self.transform_pair(distorted_image, referenced_image)

        row = self.scores_df.iloc[video_idx]
        score = row[wandb.config.subjective_score_type]
        return distorted_image, referenced_image, score, video_idx
    
    def get_scene_indices(self):
        scene_indices = {}
        for i, row in self.scores_df.iterrows():
            scene = row['distorted_folder']
            start_idx = 0 if i == 0 else self.cumulative_frame_counts.iloc[i - 1]
            end_idx = self.cumulative_frame_counts.iloc[i]
            indices = list(range(start_idx, end_idx))
            if scene not in scene_indices:
                scene_indices[scene] = []
            scene_indices[scene].extend(indices)
        return scene_indices

# Batch creation function
def create_nerf_qa_resize_dataloader(scores_df, dir, batch_size = DEVICE_BATCH_SIZE, scene_balanced=True):
    # Create a dataset and dataloader for efficient batching
    dataset = NeRFQAResizedDataset(dir=dir, scores_df=scores_df)
    sampler = SceneBalancedSampler(dataset)
    if scene_balanced:
        dataloader = DataLoader(dataset, sampler=sampler, batch_size = batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
    else:
        dataloader = DataLoader(dataset, batch_size = batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
    return dataloader

class SceneBalancedSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.scene_indices = self.dataset.get_scene_indices()
        self.num_scenes = len(self.scene_indices)
        self.samples_per_scene = min(len(indices) for indices in self.scene_indices.values())
        self.num_samples = self.num_scenes * self.samples_per_scene

    def __iter__(self):
        indices = []
        for scene_indices in self.scene_indices.values():
            N = len(scene_indices)
            new_indices = torch.tensor(scene_indices)
            new_indices = new_indices[torch.randperm(N)]
            new_indices = new_indices[:self.samples_per_scene].tolist()
            indices.extend(new_indices)
        indices = torch.tensor(indices)[torch.randperm(len(indices))].tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples
