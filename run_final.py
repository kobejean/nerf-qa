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
from nerf_qa.DISTS_pytorch.DISTS_pt_original import prepare_image
from nerf_qa.data import create_test2_dataloader, create_nerf_qa_resize_dataloader
from nerf_qa.logger import MetricCollectionLogger
from nerf_qa.settings_fr import DEVICE_BATCH_SIZE
from nerf_qa.model_stats import NeRFQAModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEBUG = True

#%%
DATA_DIR = "/home/ccl/Datasets/NeRF-QA"
SCORE_FILE = path.join(DATA_DIR, "NeRF_VQA_MOS.csv")
TEST_DATA_DIR = "/home/ccl/Datasets/Test_2-datasets"
TEST_SCORE_FILE = path.join(TEST_DATA_DIR, "scores_new.csv")

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
    parser.add_argument('--lr', type=float, default=1e-6, help='Random seed.')
    parser.add_argument('--beta1', type=float, default=0.9, help='Random seed.')
    parser.add_argument('--beta2', type=float, default=0.999, help='Random seed.')
    parser.add_argument('--eps', type=float, default=1e-7, help='Random seed.')
    parser.add_argument('--optimizer', type=str, default='adam', help='Random seed.')
    parser.add_argument('--project_weights', type=str, default='False', help='Random seed.')
    parser.add_argument('--mode', type=str, default='normal', help='Random seed.')
    parser.add_argument('--gamma', type=float, default=0.95, help='Random seed.')
    # parser.add_argument('--warmup_steps', type=int, default=0, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100, help='Random seed.')

    # Parse arguments
    args = parser.parse_args()

    # Read the CSV file
    scores_df = pd.read_csv(SCORE_FILE)
    scores_df['scene'] = scores_df['reference_filename'].str.replace('_reference.mp4', '', regex=False)

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


    config = {
        # "epochs": epochs,
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
    run = wandb.init(project='nerf-qa-2', name=exp_name, config=config)
    config = wandb.config



    mse_fn = nn.MSELoss(reduction='none')
    loss_fn = nn.L1Loss(reduction='none')

    # Specify the number of splits
    # n_splits = 4
    # gkf = GroupKFold(n_splits=n_splits)
    # groups = scores_df['scene']
    step = 0



    train_df = scores_df
    train_dataloader = create_nerf_qa_resize_dataloader(train_df, dir=DATA_DIR, batch_size=config.batch_size)
    train_size = len(train_dataloader)

    test_df = pd.read_csv(TEST_SCORE_FILE)
    test_df['scene'] = test_df['reference_folder'].str.replace('gt_', '', regex=False)
    test_balanced_dataloader = create_test2_dataloader(test_df, dir=TEST_DATA_DIR, batch_size=config.batch_size, in_memory=False, scene_balanced=True)
    test_size = len(test_balanced_dataloader)
    test_epochs = wandb.config.epochs


    test_logger = MetricCollectionLogger('Test Metrics Dict')
    train_logger = MetricCollectionLogger(f'Train Metrics Dict')

    model = NeRFQAModel(train_df=train_df, mode=config.mode).to(device)

    optimizer = optim.Adam(model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )
    iterations_per_epoch = train_size
    decay_step = 1000  # Every 1000 iterations

    # Calculate the decay rate to halve the learning rate every 1000 iterations
    gamma = 0.5 ** (iterations_per_epoch / decay_step)
    gamma = 1.0

    # Create the scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        


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
    def create_test_dataloader(row, dir):
        # Create a dataset and dataloader for efficient batching
        dataset = Test2Dataset(row, dir)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn = recursive_collate)
        return dataloader 
    

    batch_step = 0

    def test(model, test_df):
        # Test step
        model.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing..."):
                frames_data = create_test_dataloader(row, TEST_DATA_DIR)
                for ref, render in frames_data:
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
        return results_df
    
    def train_epoch(epoch, model, train_dataloader, train_size):
        global step
        # Train step
        model.train()
        

        for dist,ref,score,i in tqdm(train_dataloader, total=train_size, desc="Training..."): 
            if epoch == 0:
                warmup_lr_init_lr = config.lr * 1e-4
                warmup_lr = warmup_lr_init_lr + step * (config.lr - warmup_lr_init_lr) / train_size
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr  
            elif epoch == 1:
                scheduler.last_epoch = epoch          
            wandb.log({ f'Optimizer/lr': optimizer.param_groups[0]['lr'] }, step = step)
            optimizer.zero_grad(set_to_none=True)  # Zero the gradients after updating

            predicted_score, dists_score = model(dist.to(device),ref.to(device))
            target_score = score.to(device).float()
            
            # Compute loss
            loss = loss_fn(predicted_score, target_score)
            step += 1

            # Store metrics in logger
            scene_ids =  train_df['scene'].iloc[i.numpy()].values
            video_ids =  train_df['distorted_filename'].iloc[i.numpy()].values
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

        scheduler.step()



    for epoch in range(test_epochs):
        print(f"Epoch {epoch+1}/{test_epochs}")
        train_epoch(epoch, model, train_dataloader, train_size)

        results_df = test(model, test_df)

    results_df.to_csv('results.csv')
    torch.save(model, f'model.pth')

    # Create and log an artifact for the results
    results_artifact = wandb.Artifact('results', type='dataset')
    results_artifact.add_file('results.csv')
    wandb.log_artifact(results_artifact)

    # Create and log an artifact for the model
    model_artifact = wandb.Artifact('model', type='model')
    model_artifact.add_file(f'model.pth')
    wandb.log_artifact(model_artifact)

    wandb.finish()