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
from scipy.optimize import curve_fit

# data 
import pandas as pd
from tqdm import tqdm

# local
from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS
from nerf_qa.data import create_test2_dataloader, create_test_video_dataloader, create_large_qa_dataloader
from nerf_qa.logger import MetricCollectionLogger
from nerf_qa.settings_fr import DEVICE_BATCH_SIZE
from nerf_qa.model import NeRFQAModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%%
DATA_DIR = "/home/ccl/Datasets/Test_2-datasets"
SCORE_FILE = path.join(DATA_DIR, "scores_new.csv")
VAL_DATA_DIR = "/home/ccl/Datasets/NeRF-QA-Large-1"
VAL_SCORE_FILE = path.join(VAL_DATA_DIR, "scores.csv")
TEST_DATA_DIR = "/home/ccl/Datasets/NeRF-QA"
TEST_SCORE_FILE = path.join(TEST_DATA_DIR, "NeRF_VQA_MOS.csv")

import argparse
import wandb
import multiprocessing as mp

if __name__ == '__main__':
    # Set the start method for multiprocessing
    mp.set_start_method('spawn', force=True)

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Initialize a new run with wandb with custom configurations.')

    # Basic configurations
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Random seed.')
    parser.add_argument('--beta1', type=float, default=0.9, help='Random seed.')
    parser.add_argument('--beta2', type=float, default=0.999, help='Random seed.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Random seed.')
    parser.add_argument('--momentum_decay', type=float, default=0.004, help='Random seed.')
    parser.add_argument('--eps', type=float, default=1e-7, help='Random seed.')
    parser.add_argument('--linear_layer_lr', type=float, default=1e-3, help='Random seed.')
    parser.add_argument('--init_scene_type_bias_weight', type=float, default=0.5, help='Random seed.')
    parser.add_argument('--scene_type_bias_weight_loss_coef', type=float, default=0.1, help='Random seed.')
    parser.add_argument('--optimizer', type=str, default='adam', help='Random seed.')

    # Parse arguments
    args = parser.parse_args()


    # Read the CSV file
    scores_df = pd.read_csv(SCORE_FILE)
    scores_df['scene'] = scores_df['reference_folder'].str.replace('gt_', '', regex=False)

    # Lists of scene IDs
    real_scene_ids = ['train', 'm60', 'playground', 'truck', 'fortress', 'horns', 'trex', 'room']
    synth_scene_ids = ['ship', 'lego', 'drums', 'ficus', 'hotdog', 'materials', 'mic']

    # Function to determine the scene type
    def get_scene_type(scene):
        if scene in real_scene_ids:
            return 'real'
        elif scene in synth_scene_ids:
            return 'synthetic'
        else:
            return 'unknown'

    # Apply the function to create the 'scene_type' column
    scores_df['scene_type'] = scores_df['scene'].apply(get_scene_type)

    val_df = pd.read_csv(VAL_SCORE_FILE)
    # filter test
    val_scenes = ['ship', 'lego', 'drums', 'ficus', 'train', 'm60', 'playground', 'truck'] #+ ['room', 'hotdog', 'trex', 'chair']
    train_df = scores_df[~scores_df['scene'].isin(val_scenes)].reset_index() # + ['trex', 'horns']
    val_df = val_df[val_df['scene'].isin(val_scenes)].reset_index()

    test_df = pd.read_csv(TEST_SCORE_FILE)
    test_df['scene'] = test_df['reference_filename'].str.replace('_reference.mp4', '', regex=False)
    test_size = test_df.shape[0]

    def linear_func(x, a, b):
        return a * x + b


    # Apply the adjustment for each group and get the adjusted DISTS values

    group_x = train_df['DISTS']
    group_y = train_df['MOS']

    # Perform linear regression
    params, _ = curve_fit(linear_func, group_x, group_y)

    # Extract the parameters
    a, b = params

    train_df['DISTS_adjusted'] = (group_y - b) / a
    train_df['DISTS_a'] = a
    train_df['DISTS_b'] = b

    def adjust_dists(group):
        group_x = group['DISTS']
        group_y = group['MOS']
        
        # Perform linear regression
        params, _ = curve_fit(linear_func, group_x, group_y)
        
        # Extract the parameters
        a, b = params
        
        group['DISTS_scene_adjusted'] = (group_y - b) / a
        group['DISTS_scene_a'] = a
        group['DISTS_scene_b'] = b
        
        return group

    # Apply the adjustment for each group and get the adjusted DISTS values
    adjusted_df = train_df.groupby('scene').apply(adjust_dists).reset_index(drop=True)
    train_df['DISTS_scene_adjusted'] = adjusted_df['DISTS_scene_adjusted']
    train_df['DISTS_scene_a'] = adjusted_df['DISTS_scene_a']
    train_df['DISTS_scene_b'] = adjusted_df['DISTS_scene_b']

    def adjust_dists(group):
        group_x = group['DISTS']
        group_y = group['MOS']
        
        # Perform linear regression
        params, _ = curve_fit(linear_func, group_x, group_y)
        
        # Extract the parameters
        a, b = params
        
        group['DISTS_scene_type_adjusted'] = (group_y - b) / a
        group['DISTS_scene_type_a'] = a
        group['DISTS_scene_type_b'] = b
        
        return group

    # Apply the adjustment for each group and get the adjusted DISTS values
    adjusted_df = train_df.groupby('scene_type').apply(adjust_dists).reset_index(drop=True)
    train_df['DISTS_scene_type_adjusted'] = adjusted_df['DISTS_scene_type_adjusted']
    train_df['DISTS_scene_type_a'] = adjusted_df['DISTS_scene_type_a']
    train_df['DISTS_scene_type_b'] = adjusted_df['DISTS_scene_type_b']

    train_logger = MetricCollectionLogger('Train Metrics Dict')
    val_logger = MetricCollectionLogger('Val Metrics Dict')
    test_logger = MetricCollectionLogger('Test Metrics Dict')

    train_dataloader = create_test2_dataloader(train_df, dir=DATA_DIR, batch_size=DEVICE_BATCH_SIZE, in_memory=True)
    # val_dataloader = create_test2_dataloader(val_df, dir=DATA_DIR)
    val_dataloader = create_large_qa_dataloader(val_df, dir=VAL_DATA_DIR, resize=True, batch_size=DEVICE_BATCH_SIZE)
    train_size = len(train_dataloader)
    val_size = len(val_dataloader)

    epochs = 250
    config = {
        "epochs": epochs,
        # "lr": 5e-5,
        # "beta1": 0.99,
        # "beta2": 0.9999,
        # "eps": 1e-7,
        "batch_size": DEVICE_BATCH_SIZE,
        "resize": True
    }     
    config.update(vars(args))


    #%%
    exp_name=f"l1-bs:{config['batch_size']}-lr:{config['lr']:.0e}-b1:{config['beta1']:.2f}-b2:{config['beta2']:.3f}"

    # Initialize wandb with the parsed arguments, further simplifying parameter names
    wandb.init(project='nerf-qa', name=exp_name, config=config)
    config = wandb.config

    mse_fn = nn.MSELoss(reduction='none')
    loss_fn = nn.L1Loss(reduction='none')



    # Reset model and optimizer for each fold (if you want to start fresh for each fold)
    model = NeRFQAModel(train_df=train_df).to(device)
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(model.get_param_lr(),
            lr=config.lr, momentum=0.0)
    elif config.optimizer == 'sgd_momentum':
        optimizer = optim.SGD(model.get_param_lr(),
            lr=config.lr, momentum=config.momentum)
    elif config.optimizer == 'nadam':
        optimizer = optim.NAdam(model.get_param_lr(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            momentum_decay=config.momentum_decay
        )
    else:
        optimizer = optim.Adam(model.get_param_lr(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps
        )


    # Training loop
    step = 0
    for epoch in range(wandb.config.epochs):
        print(f"Epoch {epoch+1}/{wandb.config.epochs}")

        # Train step
        model.train()  # Set model to training mode

        for index, (dist,ref,score,i) in tqdm(enumerate(train_dataloader, 1), total=train_size, desc="Training..."):  # Start index from 1 for easier modulus operation            
            optimizer.zero_grad()  # Zero the gradients after updating

            # Load scores
            # scene_type = train_df['scene_type'].iloc[i.numpy()].values
            predicted_score = model(dist.to(device),ref.to(device), scene_type=None)
            target_score = score.to(device).float()
            
            # Compute loss
            loss = loss_fn(predicted_score, target_score)
            step += score.shape[0]

            # Store metrics in logger
            scene_ids =  train_df['scene'].iloc[i.numpy()].values
            video_ids =  train_df['distorted_folder'].iloc[i.numpy()].values
            train_logger.add_entries(
                {
                'loss': loss.detach().cpu(),
                'mse': mse_fn(predicted_score, target_score).detach().cpu(),
                # 'mos': score,
                # 'pred_score': predicted_score.detach().cpu(),
            }, video_ids = video_ids, scene_ids = scene_ids)

            # Accumulate gradients
            loss = loss.mean() #+ wandb.config.scene_type_bias_weight_loss_coef * (model.scene_type_bias_weight*model.scene_type_bias_weight)
            loss.backward()

            # Log accumulated train metrics
            train_logger.log_summary(step)
            # wandb.log({ 
            #     "Model/scene_bias_weight": model.scene_type_bias_weight.detach().cpu(),
            #     "Model/dists_weight": model.dists_weight.detach().cpu(),
            #     "Model/dists_bias": model.dists_bias.detach().cpu(),
            #     "Model/dists_weight_0": model.dists_scene_type_weight[0].detach().cpu(),
            #     "Model/dists_bias_0": model.dists_scene_type_bias[0].detach().cpu(),
            #     "Model/dists_weight_1": model.dists_scene_type_weight[1].detach().cpu(),
            #     "Model/dists_bias_1": model.dists_scene_type_bias[1].detach().cpu(),
            # }, step=step)
            # wandb.log({ 
            #     "Model/grad/scene_bias_weight": model.scene_type_bias_weight.grad,
            #     "Model/grad/dists_weight": model.dists_weight.grad,
            #     "Model/grad/dists_bias": model.dists_bias.grad,
            # }, step=step)
            
            # Update parameters every batches_per_step steps or on the last iteration
            optimizer.step()

        if (epoch+1) % 5 == 0:
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
                wandb.log({ 
                    "Model/dists_weight/alpha": wandb.Histogram(model.dists_model.alpha.detach().cpu()),
                    "Model/dists_weight/beta": wandb.Histogram(model.dists_model.beta.detach().cpu()),
                }, step=step)
            
        if (epoch+1) in [100, 150, 200, 250]:
            # Test step
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                for index, row in tqdm(test_df.iterrows(), total=test_size, desc="Testing..."):
                    # Load frames
                    dataloader = create_test_video_dataloader(row, dir=TEST_DATA_DIR, resize=config.resize, keep_aspect_ratio=True, batch_size=DEVICE_BATCH_SIZE)
                    
                    # Compute score
                    predicted_score = model.forward_dataloader(dataloader)
                    target_score = torch.tensor(row['MOS'], device=device, dtype=torch.float32)
                
                    # Store metrics in logger
                    video_ids = row['distorted_filename']
                    scene_ids = row['scene']
                    test_logger.add_entries({
                        'mse': mse_fn(predicted_score, target_score).detach().cpu(),
                        'mos': row['MOS'],
                        'pred_score': predicted_score.detach().cpu(),
                    }, video_ids=video_ids, scene_ids=scene_ids)

                # Log results
                test_logger.log_summary(step)

    wandb.finish()