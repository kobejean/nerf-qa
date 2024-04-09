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
from sklearn.model_selection import GroupKFold
from scipy.optimize import curve_fit
import math

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

DEBUG = True

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
import schedulefree

if __name__ == '__main__':
    # Set the start method for multiprocessing
    mp.set_start_method('spawn', force=True)

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Initialize a new run with wandb with custom configurations.')

    # Basic configurations
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Random seed.')
    parser.add_argument('--beta1', type=float, default=0.9, help='Random seed.')
    parser.add_argument('--beta2', type=float, default=0.9995, help='Random seed.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Random seed.')
    parser.add_argument('--momentum_decay', type=float, default=0.004, help='Random seed.')
    parser.add_argument('--eps', type=float, default=1e-7, help='Random seed.')
    parser.add_argument('--linear_layer_lr', type=float, default=1e-5, help='Random seed.')
    parser.add_argument('--cnn_layer_lr', type=float, default=1e-3, help='Random seed.')
    parser.add_argument('--init_scene_type_bias_weight', type=float, default=0.5, help='Random seed.')
    parser.add_argument('--scene_type_bias_weight_loss_coef', type=float, default=0.1, help='Random seed.')
    parser.add_argument('--optimizer', type=str, default='adam', help='Random seed.')
    parser.add_argument('--project_weights', type=str, default='True', help='Random seed.')
    parser.add_argument('--gamma', type=float, default=0.95, help='Random seed.')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Random seed.')

    # Parse arguments
    args = parser.parse_args()


    # Read the CSV file
    scores_df = pd.read_csv(SCORE_FILE)
    scores_df['scene'] = scores_df['reference_folder'].str.replace('gt_', '', regex=False)

    # Lists of scene IDs
    real_scene_ids = ['train', 'm60', 'playground', 'truck', 'fortress', 'horns', 'trex', 'room']
    synth_scene_ids = ['ship', 'lego', 'drums', 'ficus', 'hotdog', 'materials', 'mic', 'chair']

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

    # val_df = pd.read_csv(VAL_SCORE_FILE)
    # # filter test
    test_scenes = ['ship', 'lego', 'drums', 'ficus', 'train', 'm60', 'playground', 'truck'] #+ ['room', 'hotdog', 'trex', 'chair']
    scores_df = scores_df[~scores_df['scene'].isin(test_scenes)].reset_index() # + ['trex', 'horns']

    epochs = 50
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
    exp_name=f"l1-bs:{config['batch_size']}-lr:{config['lr']:.0e}-b1:{config['beta1']:.3f}-b2:{config['beta2']:.4f}"

    # Initialize wandb with the parsed arguments, further simplifying parameter names
    wandb.init(project='nerf-qa', name=exp_name, config=config)
    config = wandb.config

    mse_fn = nn.MSELoss(reduction='none')
    loss_fn = nn.L1Loss(reduction='none')

    # Specify the number of splits
    n_splits = 4
    gkf = GroupKFold(n_splits=n_splits)
    groups = scores_df['scene']
    step = 0

    cv_correlations = []
    cv_scene_mins = []
    cv_last_mses = []

    # Create splits
    for fold, (train_idx, val_idx) in enumerate(gkf.split(scores_df, groups=groups)):
        train_df = scores_df.iloc[train_idx].reset_index(drop=True)
        val_df = scores_df.iloc[val_idx].reset_index(drop=True)

        train_logger = MetricCollectionLogger(f'Train Metrics Dict/fold_{fold}')
        val_logger = MetricCollectionLogger(f'Val Metrics Dict/fold_{fold}')

        train_dataloader = create_test2_dataloader(train_df, dir=DATA_DIR, batch_size=DEVICE_BATCH_SIZE, in_memory=False)
        val_dataloader = create_test2_dataloader(val_df, dir=DATA_DIR, batch_size=DEVICE_BATCH_SIZE, in_memory=False, scene_balanced=False)
        train_size = len(train_dataloader)
        val_size = len(val_dataloader)


        # Reset model and optimizer for each fold (if you want to start fresh for each fold)
        model = NeRFQAModel(train_df=train_df).to(device)
        # optimizer = optim.Adam(model.parameters(),
        #     lr=config.lr,
        #     betas=(config.beta1, config.beta2),
        #     eps=config.eps
        # )
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(),                
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            warmup_steps=config.warmup_steps,
        )
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    


        # Training loop
        for epoch in range(wandb.config.epochs):
            print(f"Epoch {epoch+1}/{wandb.config.epochs}")

            # Train step
            model.train()  # Set model to training mode
            optimizer.train()

            for dist,ref,score,i in tqdm(train_dataloader, total=train_size, desc="Training..."):  # Start index from 1 for easier modulus operation            
                optimizer.zero_grad()  # Zero the gradients after updating

                # Load scores
                predicted_score = model(dist.to(device),ref.to(device))
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
                }, video_ids = video_ids, scene_ids = scene_ids)

                # Accumulate gradients
                loss = loss.mean()
                loss.backward()

                # Log accumulated train metrics
                train_logger.log_summary(step)
                
                # Update parameters every batches_per_step steps or on the last iteration
                optimizer.step()
                if config.project_weights == 'True':
                    model.dists_model.project_weights()
            # scheduler.step()
            if (epoch+1) % 10 == 0:
                # Validation step
                model.eval()  # Set model to evaluation mode
                optimizer.eval()
                with torch.no_grad():
                    for dist, ref, score, i in tqdm(val_dataloader, total=val_size, desc="Validating..."):
                        # Compute score
                        predicted_score = model(dist.to(device), ref.to(device))
                        target_score = score.to(device).float()

                        # Compute loss
                        loss = loss_fn(predicted_score, target_score)
                        
                        # Store metrics in logger
                        scene_ids = val_df['scene'].iloc[i.numpy()].values
                        video_ids = val_df['distorted_folder'].iloc[i.numpy()].values
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
                        "Model/dists_weight/alpha_min": torch.min(model.dists_model.alpha),
                        "Model/dists_weight/beta_min": torch.min(model.dists_model.beta),
                    }, step=step)
        
        cv_correlations.append(val_logger.last_correlations)
        cv_scene_mins.append(val_logger.last_scene_min)
        cv_last_mses.append(val_logger.last_mse)

    cv_correlations_concat = {}
    cv_scene_mins_concat = {}

    # Loop through each dictionary in the list
    for scores in cv_correlations:
        for key, value in scores.items():
            if key in cv_correlations_concat:
                cv_correlations_concat[key].append(value)
            else:
                cv_correlations_concat[key] = [value]
    
    # Loop through each dictionary in the list
    for scores in cv_scene_mins:
        for key, value in scores.items():
            if key in cv_scene_mins_concat:
                cv_scene_mins_concat[key].append(value)
            else:
                cv_scene_mins_concat[key] = [value]


    for key, value in cv_correlations_concat.items():
        wandb.log({ 
            f"Cross-Val Metrics Dict/correlations/mean_{key}": np.mean(value),
            f"Cross-Val Metrics Dict/correlations/std_{key}": np.std(value),
        }, step=step)

    for key, value in cv_scene_mins_concat.items():
        wandb.log({ 
            f"Cross-Val Metrics Dict/correlations/scene_min/mean_{key}": np.mean(value),
            f"Cross-Val Metrics Dict/correlations/scene_min/std_{key}": np.std(value),
        }, step=step)
    wandb.log({ 
        f"Cross-Val Metrics Dict/mean_mse": np.mean(cv_last_mses),
        f"Cross-Val Metrics Dict/std_mse": np.std(cv_last_mses),
    }, step=step)


    del train_dataloader
    del val_dataloader
    train_df = scores_df
    train_dataloader = create_test2_dataloader(train_df, dir=DATA_DIR, batch_size=DEVICE_BATCH_SIZE, in_memory=False)
    train_size = len(train_dataloader)

    test_df = pd.read_csv(TEST_SCORE_FILE)
    test_df['scene'] = test_df['reference_filename'].str.replace('_reference.mp4', '', regex=False)
    test_size = test_df.shape[0]

    test_logger = MetricCollectionLogger('Test Metrics Dict')
    train_logger = MetricCollectionLogger(f'Train Metrics Dict')

    model = NeRFQAModel(train_df=train_df).to(device)
    # optimizer = optim.Adam(model.get_param_lr(),
    #     lr=config.lr,
    #     betas=(config.beta1, config.beta2),
    #     eps=config.eps
    # )
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(),                
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        warmup_steps=config.warmup_steps,
    )
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    
    for epoch in range(wandb.config.epochs):
        print(f"Epoch {epoch+1}/{wandb.config.epochs}")

        # Train step
        model.train()  # Set model to training mode
        optimizer.train()

        for dist,ref,score,i in tqdm(train_dataloader, total=train_size, desc="Training..."):  # Start index from 1 for easier modulus operation            
            optimizer.zero_grad()  # Zero the gradients after updating

            # Load scores
            # scene_type = train_df['scene_type'].iloc[i.numpy()].values
            predicted_score = model(dist.to(device),ref.to(device))
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
            loss = loss.mean()
            loss.backward()

            # Log accumulated train metrics
            train_logger.log_summary(step)
            
            # Update parameters every batches_per_step steps or on the last iteration
            optimizer.step()
            if config.project_weights == 'True':
                model.dists_model.project_weights()
        # scheduler.step()

    # Test step
    model.eval()  # Set model to evaluation mode
    optimizer.eval()
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
                'dmos': row['DMOS'],
                'pred_score': predicted_score.detach().cpu(),
            }, video_ids=video_ids, scene_ids=scene_ids)

        # Log results
        results_df = test_logger.video_metrics_df()
        results_df.to_csv('results.csv')
        test_logger.log_summary(step)

    torch.save(model, f'{exp_name}.pth')

    wandb.finish()