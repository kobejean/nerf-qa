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
from nerf_qa.data import NerfNRQADataset
from nerf_qa.logger import MetricCollectionLogger
from nerf_qa.settings import DEVICE_BATCH_SIZE
from nerf_qa.model_nr_v2 import NRModel
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
        frame = transforms.ToTensor()(frame)
        render_256 = F.interpolate(frame.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        render_224 = F.interpolate(frame.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        render = { "256x256": render_256, "224x224": render_224 }
        frames.append(render)
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
    
# Batch creation function
def create_test_video_dataloader(row, dir, resize=True):
    #ref_dir = path.join(dir, "Reference")
    syn_dir = path.join(dir, "NeRF-QA_videos")
    dist_video_path = path.join(syn_dir, row['distorted_filename'])
    #ref_video_path = path.join(ref_dir, row['reference_filename'])
    #ref = load_video_frames(ref_video_path, resize=resize)
    dist = load_video_frames(dist_video_path, resize=resize)
    # Create a dataset and dataloader for efficient batching
    dataset = CustomDataset(dist)
    dataloader = DataLoader(dataset, batch_size=DEVICE_BATCH_SIZE, shuffle=False, collate_fn = recursive_collate)
    return dataloader  
if __name__ == '__main__':
    # Set the start method for multiprocessing
    mp.set_start_method('spawn', force=True)


    #%%
    DATA_DIR = "/home/ccl/Datasets/NeRF-QA-Large-1"
    SCORE_FILE = path.join(DATA_DIR, "scores.csv")
    TEST_DATA_DIR = "/home/ccl/Datasets/NeRF-QA"
    TEST_SCORE_FILE = path.join(TEST_DATA_DIR, "NeRF_VQA_MOS.csv")


    # Set up argument parser
    parser = argparse.ArgumentParser(description='Initialize a new run with wandb with custom configurations.')

    # Basic configurations
    parser.add_argument('--refine_up_depth', type=int, default=2, help='Random seed.')
    parser.add_argument('--transformer_decoder_depth', type=int, default=1, help='Random seed.')
    parser.add_argument('--refine_scale', type=float, default=0.1, help='Random seed.')
    parser.add_argument('--score_reg_scale', type=float, default=0.05, help='Random seed.')
    parser.add_argument('--dists_pref2ref_coeff', type=float, default=0.5, help='Random seed.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Random seed.')

    # Parse arguments
    args = parser.parse_args()

    epoch_size = 3290
    epochs = 8
    config = {
        "epochs": epochs,
        "loader_num_workers": 4,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-7,
        "batch_size": DEVICE_BATCH_SIZE,
    }     
    config.update(vars(args))

    exp_name=f"l1-bs:{config['batch_size']}-lr:{config['lr']:.0e}-b1:{config['beta1']:.2f}-b2:{config['beta2']:.3f}"

    # Initialize wandb with the parsed arguments, further simplifying parameter names
    wandb.init(project='nerf-nr-qa', name=exp_name, config=config)
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATA_DIR = "/home/ccl/Datasets/NeRF-NR-QA/"  # Specify the path to your DATA_DIR



    # CSV file 
    scores_df = pd.read_csv("/home/ccl/Datasets/NeRF-NR-QA/output.csv")
    val_scenes = ['nerfstudio_plane', 'nerfstudio_stump', 'mipnerf360_garden', 'mipnerf360_stump']
    train_df = scores_df[~scores_df['scene'].isin(val_scenes)].reset_index() # + ['trex', 'horns']
    val_df = scores_df[scores_df['scene'].isin(val_scenes)].reset_index()
    black_list_val_methods = [
        'instant-ngp-10', 'instant-ngp-20', 'instant-ngp-50', 'instant-ngp-100', 'instant-ngp-200', 'instant-ngp-500', 'instant-ngp-1000', 'instant-ngp-2000', 'instant-ngp-5000', 'instant-ngp-10000', 'instant-ngp-20000',
        'nerfacto-10', 'nerfacto-20', 'nerfacto-50', 'nerfacto-100', 'nerfacto-200', 'nerfacto-500', 'nerfacto-1000', 'nerfacto-2000', 'nerfacto-5000', 'nerfacto-10000', 'nerfacto-20000',
    ]
    val_df = val_df[~val_df['method'].isin(black_list_val_methods)].reset_index()

    test_df = pd.read_csv(TEST_SCORE_FILE)
    test_size = test_df.shape[0]

    train_dataset = NerfNRQADataset(train_df, dir = DATA_DIR, mode='gt')
    val_dataset = NerfNRQADataset(val_df, dir = DATA_DIR, mode='gt')
    
    train_dataloader = DataLoader(train_dataset, collate_fn=recursive_collate, shuffle=True, batch_size = DEVICE_BATCH_SIZE, num_workers=config.loader_num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=recursive_collate, batch_size = DEVICE_BATCH_SIZE, num_workers=config.loader_num_workers, pin_memory=True)

    

    model = NRModel(device=device, refine_up_depth=config.refine_up_depth)
    optimizer = optim.Adam(model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps
    )
    step = 0
    for epoch in range(config.epochs):

        model.train()

        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("load_data"):
            for batch in tqdm(train_dataloader):
                gt_image, render, score, render_id, frame_id = batch_to_device(batch, device)
                optimizer.zero_grad()
                with record_function("model_inference"):
                    losses = model.losses(gt_image, render, score)
                loss = losses['combined']
                loss.backward()
                optimizer.step()
                for key in losses.keys():
                    wandb.log({
                        f"Training Metrics Dict/{key}": losses[key].cpu().item()
                    }, step=step)
                step += score.shape[0]  
        # Note: The profiler is not thread-safe, so if your dataloader uses multiprocessing (num_workers > 0),
        # make sure to start the profiler after dataloader worker processes have been spawned.
        #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        model.eval()
        metrics = []
        for batch in tqdm(val_dataloader):
            gt_image, render, score, render_id, frame_id = batch_to_device(batch, device)
            with torch.no_grad():
                losses = model.losses(gt_image, render, score)
            loss = losses['combined']
            metrics.append(losses['l1'].cpu().item())
        wandb.log({
            "Validation Metrics Dict/l1": np.mean(metrics)
        }, step=step)

        # Test step
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            video_scores = []
            for index, row in tqdm(test_df.iterrows(), total=test_size, desc="Testing..."):
                # Load frames
                dataloader = create_test_video_dataloader(row, dir=TEST_DATA_DIR)
                frame_scores = []
                for batch in dataloader:
                    render = batch_to_device(batch, device)
                    pred_score = model(render)
                    frame_scores.append(pred_score.detach().cpu())

                frame_scores = np.concatenate(frame_scores, axis=0)
                video_scores.append(np.mean(frame_scores))

            video_scores = np.array(video_scores)
            corr = compute_correlations(video_scores, test_df['MOS'].values)
            corr['l1'] = np.mean(np.abs(video_scores - test_df['DISTS'].values))
            for key in corr.keys():
                metric = corr[key]
                wandb.log({
                    f"Test Metrics Dict/{key}": metric
                }, step=step)

# %%
