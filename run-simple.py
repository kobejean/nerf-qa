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

# local
from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS
from nerf_qa.data import create_large_qa_dataloader, create_test_video_dataloader
from nerf_qa.logger import MetricCollectionLogger
from nerf_qa.settings import DEVICE_BATCH_SIZE
from nerf_qa.model import NeRFQAModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%%
DATA_DIR = "/home/ccl/Datasets/NeRF-QA-Large-1"
SCORE_FILE = path.join(DATA_DIR, "scores.csv")
TEST_DATA_DIR = "/home/ccl/Datasets/NeRF-QA"
TEST_SCORE_FILE = path.join(TEST_DATA_DIR, "NeRF_VQA_MOS.csv")

import argparse
import wandb

# Set up argument parser
parser = argparse.ArgumentParser(description='Initialize a new run with wandb with custom configurations.')

# Basic configurations
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

# Parse arguments
args = parser.parse_args()

epoch_size = 3290
batches_per_step = -(epoch_size // -DEVICE_BATCH_SIZE)
epochs = 50
config = {
    "epochs": epochs,
    "batches_per_step": batches_per_step,
    "lr": 1e-4,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-7,
    "batch_size": epoch_size,
}     
config.update(vars(args))


#%%
exp_name=f"l1-fixed-lin-bs:{config['batch_size']}-lr:{config['lr']:.0e}-b1:{config['beta1']:.2f}-b2:{config['beta2']:.2f}"

# Initialize wandb with the parsed arguments, further simplifying parameter names
wandb.init(project='nerf-qa-test', name=exp_name, config=config)
config = wandb.config

# Read the CSV file
scores_df = pd.read_csv(SCORE_FILE)
# filter test
test_scenes = ['ship', 'lego', 'drums', 'ficus', 'train', 'm60', 'playground', 'truck']
train_df = scores_df[~scores_df['scene'].isin(test_scenes)].reset_index() # + ['trex', 'horns']
val_df = scores_df[scores_df['scene'].isin(test_scenes)].reset_index()

test_df = pd.read_csv(TEST_SCORE_FILE)
test_size = test_df.shape[0]

mse_fn = nn.MSELoss(reduction='none')
loss_fn = nn.L1Loss(reduction='none')


step = 0

# Reset model and optimizer for each fold (if you want to start fresh for each fold)
model = NeRFQAModel(train_df=train_df).to(device)
optimizer = optim.Adam(model.parameters(),
    lr=config.lr,
    betas=(config.beta1, config.beta2),
    eps=config.eps
)


train_logger = MetricCollectionLogger('Train Metrics Dict')
val_logger = MetricCollectionLogger('Val Metrics Dict')
test_logger = MetricCollectionLogger('Test Metrics Dict')

train_dataloader = create_large_qa_dataloader(train_df, dir=DATA_DIR)
val_dataloader = create_large_qa_dataloader(val_df, dir=DATA_DIR)
train_size = len(train_dataloader)
val_size = len(val_dataloader)

# Training loop
for epoch in range(wandb.config.epochs):
    print(f"Epoch {epoch+1}/{wandb.config.epochs}")

    # Test step
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():

        for index, row in tqdm(test_df.iterrows(), total=test_size, desc="Testing..."):
            # Load frames
            dataloader = create_test_video_dataloader(row, dir=TEST_DATA_DIR)
            
            # Compute score
            predicted_score = model.forward_dataloader(dataloader)
            target_score = torch.tensor(row['MOS'], device=device, dtype=torch.float32)
        
            # Store metrics in logger
            video_ids = row['distorted_filename']
            scene_ids = row['reference_filename']
            test_logger.add_entries({
                'mse': mse_fn(predicted_score, target_score).detach().cpu(),
                'mos': row['MOS'],
                'pred_score': predicted_score.detach().cpu(),
            }, video_ids=video_ids, scene_ids=scene_ids)

    # Log results
    test_logger.log_summary(step)

    # Train step
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Initialize gradients to zero at the start of each epoch
    weight_sum = 0

    for index, (dist,ref,score,i) in tqdm(enumerate(train_dataloader, 1), total=train_size, desc="Training..."):  # Start index from 1 for easier modulus operation            
        # Load scores
        predicted_score = model(dist.to(device),ref.to(device))
        target_score = score.to(device).float()
        
        # Compute loss
        loss = loss_fn(predicted_score, target_score)
        weights = 1 / torch.tensor(train_df['frame_count'].iloc[i.numpy()].values, device=device, dtype=torch.float32)
        step += target_score.shape[0]

        # Store metrics in logger
        scene_ids =  train_df['scene'].iloc[i.numpy()].values
        video_ids =  train_df['distorted_filename'].iloc[i.numpy()].values
        train_logger.add_entries(
            {
            'loss': loss.detach().cpu(),
            'mse': mse_fn(predicted_score, target_score).detach().cpu(),
            'mos': score,
            'pred_score': predicted_score.detach().cpu(),
        }, video_ids = video_ids, scene_ids = scene_ids)

        # Accumulate gradients
        loss = torch.dot(loss, weights)
        loss.backward()
        weight_sum += weights.sum().item()

    # Scale gradients
    for param in model.parameters():
        if param.grad is not None:
            param.grad /= weight_sum

    # Log accumulated train metrics
    train_logger.log_summary(step)
    
    # Update parameters every batches_per_step steps or on the last iteration
    optimizer.step()
    optimizer.zero_grad()  # Zero the gradients after updating


    # Validation step
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for dist, ref, score, i in tqdm(val_dataloader, total=val_size, desc="Validating..."):
            # Compute score
            predicted_score = model(dist.to(device), ref.to(device))
            target_score = score.to(device).float()
        
            # Compute loss
            loss = loss_fn(predicted_score, target_score)
            
            # Store metrics in logger
            scene_ids = val_df['scene'].iloc[i.numpy()].values
            video_ids = val_df['distorted_filename'].iloc[i.numpy()].values
            val_logger.add_entries(
                {
                'loss': loss.detach().cpu(),
                'mse': mse_fn(predicted_score, target_score).detach().cpu(),
                'mos': score,
                'pred_score': predicted_score.detach().cpu(),
            }, video_ids = video_ids, scene_ids = scene_ids)

        # Log accumulated metrics
        val_logger.log_summary(step)

wandb.finish()