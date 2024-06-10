#%%
# system level
import os
from os import path
import sys
import argparse
from torchvision import models,transforms


# deep learning
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from scipy.optimize import curve_fit
from torch.utils.data import Dataset, DataLoader, Sampler
import math

# data 
import pandas as pd
from tqdm import tqdm
from PIL import Image

# local
from nerf_qa.data_fr import create_test2_dataloader, create_nerf_qa_resize_dataloader
from nerf_qa.logger import MetricCollectionLogger
from nerf_qa.settings_fr import DEVICE_BATCH_SIZE
from nerf_qa.model_stats import NeRFQAModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    parser.add_argument('--lr', type=float, default=1e-4, help='Random seed.')
    parser.add_argument('--beta1', type=float, default=0.9, help='Random seed.')
    parser.add_argument('--beta2', type=float, default=0.999, help='Random seed.')
    parser.add_argument('--entropy_loss_coeff', type=float, default=0.0, help='Random seed.')
    parser.add_argument('--eps', type=float, default=1e-7, help='Random seed.')
    parser.add_argument('--optimizer', type=str, default='adam', help='Random seed.')
    parser.add_argument('--dists_weight_norm', type=str, default='relu', help='Random seed.')
    parser.add_argument('--real_scenes_only', type=str, default='False', help='Random seed.')
    parser.add_argument('--detach_beta', type=str, default='False', help='Random seed.')
    parser.add_argument('--regression_type', type=str, default='linear', help='Random seed.')
    parser.add_argument('--subjective_score_type', type=str, default='MOS', help='Random seed.')
    parser.add_argument('--weight_lower_bound', type=float, default=1e-4, help='Random seed.')
    parser.add_argument('--alpha_beta_ratio', type=float, default=2.0, help='Random seed.')
    parser.add_argument('--gamma', type=float, default=0.5, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=3, help='Random seed.')

    # Parse arguments
    args = parser.parse_args()


    config = {
        # "epochs": epochs,
        # "lr": 5e-5,
        # "beta1": 0.99,
        # "beta2": 0.9999,
        # "eps": 1e-7,
        "batch_size": DEVICE_BATCH_SIZE,
        "resize": False
    }     
    config.update(vars(args))
    run = wandb.init(project='nerf-qa-2-reeval', config=config)
    config = wandb.config

    model = torch.load('model_fin.pth')

    TEST_DATA_DIR = "/home/ccl/Datasets/Test_2-datasets"
    TEST_SCORE_FILE = path.join(TEST_DATA_DIR, "scores_aspect.csv")
    test_df = pd.read_csv(TEST_SCORE_FILE)
    test_df['scene'] = test_df['reference_folder'].str.replace('gt_', '', regex=False)
    test_logger = MetricCollectionLogger('Test Metrics Dict')
    train_logger = MetricCollectionLogger(f'Train Metrics Dict')

    step = 0

    # Test step
    model.eval()  # Set model to evaluation mode

    mse_fn = nn.MSELoss(reduction='none')
    loss_fn = nn.L1Loss(reduction='none')

    with torch.no_grad():
        for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing..."):
            frames_data = create_test2_dataloader(row, TEST_DATA_DIR, batch_size=32, resize=False)
            for ref, render in frames_data:
                print(render.shape)
                i = np.full(shape=render.shape[0], fill_value=index)
                # Compute score
                predicted_score, dists_score = model(render.to(device), ref.to(device))
                score = test_df['MOS'].iloc[i].values
                target_score = torch.tensor(score, device=device).float()

                # Compute loss
                loss = loss_fn(predicted_score, target_score)
                
                # Store metrics in logger
                scene_ids = test_df['scene'].iloc[i].values
                video_ids = test_df['distorted_folder'].iloc[i].values
                test_logger.add_entries(
                    {
                    'loss': loss.detach().cpu(),
                    'mse': mse_fn(predicted_score, target_score).detach().cpu(),
                    'mos': score,
                    'pred_score': dists_score.detach().cpu(),
                }, video_ids = video_ids, scene_ids = scene_ids)


        results_df = test_logger.video_metrics_df()
        test_logger.log_summary(step)
    model.train()
    results_df.to_csv(f'results_reeval.csv')

