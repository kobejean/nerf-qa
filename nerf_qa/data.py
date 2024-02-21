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

# local
from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS, prepare_image

class CustomImageDataset(Dataset):

    def __init__(self, dir, scores_df, resize=True):
        self.ref_dir = path.join(dir, "Reference")
        self.dist_dir = path.join(dir, "NeRF-QA_videos")
        self.scores_df = scores_df
        self.resize = resize
        self.total_size = self._count_all_frames()

    def _load_video_frames(self, video_path):
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
            frame = prepare_image(frame, resize=self.resize).squeeze(0)
            frames.append(frame)
        cap.release()
        return torch.stack(frames)
    

    def _count_all_frames(self):
        def _count_frames(video_path):
            cap = cv2.VideoCapture(video_path)
            count += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        count = 0
        for i, row in self.scores_df.iterrows():
            dist_video_path = path.join(self.dist_dir, row['distorted_filename'])
            ref_video_path = path.join(self.ref_dir, row['reference_filename'])
            dist_count = _count_frames(dist_video_path)
            ref_count = _count_frames(ref_video_path)
            assert dist_count == ref_count
            count += ref_count
        return count


    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        # return image, label