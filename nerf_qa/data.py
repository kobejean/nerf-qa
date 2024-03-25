
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

class Test2DatasetVideo(Dataset):
    def __init__(self, row, dir):
        gt_dir = path.join(dir, "Reference", row['distorted_folder'])
        render_dir = path.join(dir, "Renders", row['reference_folder'])

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
        image_256 = F.interpolate(image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        image_224 = F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

        return { "256x256": image_256, "224x224": image_224 }

class Test2Dataset(Dataset):

    def __init__(self, dir, scores_df):
        self.ref_dir = ref_dir = path.join(dir, "Reference")
        self.dist_dir = dist_dir = path.join(dir, "Renders")
        self.scores_df = scores_df
        self.total_size = self.scores_df['frame_count'].sum()
        self.cumulative_frame_counts = self.scores_df['frame_count'].cumsum()
        def get_files(row, base_dir, column_name):
            folder_path = os.path.join(base_dir, row[column_name])
            file_list = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png"))]
            file_list.sort()
            return file_list
        self.scores_df['render_files'] = self.scores_df.apply(get_files, axis=1, args=(dist_dir, 'distorted_folder'))
        self.scores_df['gt_files'] = self.scores_df.apply(get_files, axis=1, args=(ref_dir, 'reference_folder'))

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
        distorted_foldername = self.scores_df.iloc[video_idx]['distorted_folder']
        referenced_foldername = self.scores_df.iloc[video_idx]['reference_folder']
        distorted_filename = self.scores_df.iloc[video_idx]['render_files'][frame_within_video]
        referenced_filename = self.scores_df.iloc[video_idx]['gt_files'][frame_within_video]

        # Construct the full paths
        distorted_path = os.path.join(self.dist_dir, distorted_foldername, distorted_filename)
        referenced_path = os.path.join(self.ref_dir, referenced_foldername, referenced_filename)

        # Load and optionally resize images
        distorted_image = prepare_image(Image.open(distorted_path).convert("RGB"), resize=True, keep_aspect_ratio=True).squeeze(0)
        referenced_image = prepare_image(Image.open(referenced_path).convert("RGB"), resize=True, keep_aspect_ratio=True).squeeze(0)

        row = self.scores_df.iloc[video_idx]
        score = row['MOS']
        return distorted_image, referenced_image, score, video_idx

# Batch creation function
def create_test2_dataloader(scores_df, dir):
    # Create a dataset and dataloader for efficient batching
    dataset = Test2Dataset(dir=dir, scores_df=scores_df)
    sampler = ComputeBatchSampler(dataset, DEVICE_BATCH_SIZE)
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    return dataloader

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
        frame = prepare_image(frame, resize=resize, keep_aspect_ratio=keep_aspect_ratio).squeeze(0)
        frames.append(frame)
    cap.release()
    return torch.stack(frames)

# Batch creation function
def create_test_video_dataloader(row, dir, resize=True, keep_aspect_ratio=False):
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
            indices.extend(torch.randperm(len(scene_indices))[:self.samples_per_scene].tolist())
        indices = torch.tensor(indices)[torch.randperm(len(indices))].tolist()
        return iter(indices)

    def __len__(self):
        return self.num_samples

#%%

class NerfNRQADataset(Dataset):
    def __init__(self, dataframe, dir = "/home/ccl/Datasets/NeRF-NR-QA/", mode='render', is_train=False, aug_crop_scale=0.8, aug_rot_deg=30.0):
        self.dir = dir
        self.df = dataframe
        self.total_frames = self.df['frame_count'].sum()
        self.frames = self.df['frame_count'].cumsum()
        self.mode = mode
        self.is_train = is_train
        self.aug_crop_scale = aug_crop_scale
        self.aug_rot_deg = aug_rot_deg

    def __len__(self):
        return self.total_frames
    
    def get_scene_indices(self):
        scene_indices = {}
        for i, row in self.df.iterrows():
            scene = row['scene']
            start_idx = 0 if i == 0 else self.frames.iloc[i - 1]
            end_idx = self.frames.iloc[i]
            indices = list(range(start_idx, end_idx))
            if scene not in scene_indices:
                scene_indices[scene] = []
            scene_indices[scene].extend(indices)
        return scene_indices

    def __getitem__(self, index):
        df_index = self.frames.searchsorted(index, side='right')
        if df_index > 0:
            frame_index = index - self.frames.iloc[df_index - 1]
        else:
            frame_index = index
        row = self.df.iloc[df_index]
        scene = row['scene']
        method = row['method']
        
        basenames = eval(row['basenames'])  # Convert string to list
        if frame_index >= len(basenames):
            print(df_index, scene, method, frame_index)
        basename = basenames[frame_index]
        dists_std = eval(row['DISTS_std'])[frame_index]  # Get DISTS score for the specific frame
        dists_mean = eval(row['DISTS_mean'])[frame_index]  # Get DISTS score for the specific frame
        render_dir = row['render_dir']

        if os.path.basename(render_dir) == 'color':
            score_map_dir = os.path.join(os.path.dirname(render_dir), 'score-map')
        else:
            score_map_dir = os.path.join(os.path.dirname(render_dir), 'gt-score-map')

        gt_dir = row['gt_dir']


        render_path = os.path.join(self.dir, render_dir, basename)
        render_image = self.load_image(render_path)
        gt_path = os.path.join(self.dir, gt_dir, basename)
        
        gt_image = self.load_image(gt_path)
        render_image, gt_image = self.transform_pair(render_image, gt_image)
        
        render_256 = F.interpolate(render_image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        render_224 = F.interpolate(render_image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        render = { "256x256": render_256, "224x224": render_224 }

        gt_image = F.interpolate(gt_image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        if self.mode == 'score-map':
            score_map_path = os.path.join(self.dir, score_map_dir, basename)
            score_map = Image.open(score_map_path)
            log_min = eval(row['score_map_log_min'])[frame_index].item()
            log_max = eval(row['score_map_log_max'])[frame_index].item()
            score_map = self.transform(score_map)
            if score_map.shape[0] == 1:
                score_map = (log_max-log_min) * score_map + log_min
            else:
                score_map[1] = (log_max-log_min) * score_map[1] + log_min
            return gt_image, render, score_map, df_index, frame_index
        return gt_image, render, torch.tensor(dists_std), torch.tensor(dists_mean), df_index, frame_index
    
    def transform_pair(self, render_image, gt_image):
        if self.is_train:
            rotation_degrees = self.aug_rot_deg  # Random rotation range in degrees
            angle = transforms.RandomRotation.get_params(degrees=(-rotation_degrees, rotation_degrees))
            render_image = TF.rotate(render_image, angle)
            gt_image = TF.rotate(gt_image, angle)

        h, w = (int(render_image.shape[1]*0.7), int(render_image.shape[2]*0.7))
        i, j = (render_image.shape[1]-h)//2, (render_image.shape[2]-w)//2
        # Crop to avoid black region due to postprocessed distortion
        render_image = TF.crop(render_image, i, j, h, w)
        gt_image = TF.crop(gt_image, i, j, h, w)


        if self.is_train:
            # Define the transformations
            crop_scale = self.aug_crop_scale
            crop_size = int(crop_scale * h), int(crop_scale * w)  
            # Apply the same transformations to both images
            i, j, h, w = transforms.RandomCrop.get_params(render_image, output_size=crop_size)
            render_image = TF.crop(render_image, i, j, h, w)
            gt_image = TF.crop(gt_image, i, j, h, w)

        return render_image, gt_image

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
        image = self.transform(image)
        return image

    def transform(self, image):
        # Apply any necessary image transformations here
        # For example, you can resize, normalize, or convert to tensor
        return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
#%%
if __name__ == "__main__":

    DATA_DIR = "/home/ccl/Datasets/NeRF-NR-QA/"  # Specify the path to your DATA_DIR

    # CSV file path
    csv_file = "/home/ccl/Datasets/NeRF-NR-QA/output.csv"
    # Read the CSV file
    scores_df = pd.read_csv(csv_file)
    dataset = NerfNRQADataset(scores_df, dir=DATA_DIR, mode='gt',is_train=True)
    batch = dataset[7000]
    to_pil = transforms.ToPILImage()
    print(batch[0].shape)
    display(to_pil(batch[0]))
    display(to_pil(batch[1]['256x256']))



# %%
