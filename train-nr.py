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
from nerf_qa.model_nr import NRModel
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
    parser.add_argument('--refine_scale', type=float, default=0.1, help='Random seed.')
    parser.add_argument('--dists_pref2ref_coeff', type=float, default=0.5, help='Random seed.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Random seed.')

    # Parse arguments
    args = parser.parse_args()

    epoch_size = 3290
    epochs = 20
    config = {
        "epochs": epochs,
        "loader_num_workers": 6,
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
