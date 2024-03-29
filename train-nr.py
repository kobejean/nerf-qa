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
from nerf_qa.data import NerfNRQADataset, SceneBalancedSampler
from nerf_qa.logger import MetricCollectionLogger
from nerf_qa.settings import DEVICE_BATCH_SIZE
from nerf_qa.model_nr_v8 import NRModel
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

class MetricAggregator:
    def __init__(self, title):
        self.metrics = []
        self.summed_metrics = {}
        self.count_metrics = {}
        self.title = title
    
    def add_metric(self, metric_dict):
        """
        Add a new set of metrics.
        
        Parameters:
        - metric_dict: A dictionary of metrics where keys are metric names and values are the metric values.
        """
        self.metrics.append(metric_dict)
        
        # Update summed_metrics and count_metrics
        for key, value in metric_dict.items():
            if key in self.summed_metrics:
                self.summed_metrics[key] += value.item()  # Assuming the values are tensor, calling item() to get the value
                self.count_metrics[key] += 1
            else:
                self.summed_metrics[key] = value.item()
                self.count_metrics[key] = 1

    def log_summary(self, step):
        """
        Calculate the average for each metric and log them using wandb.
        
        Parameters:
        - step: The current step to log the metrics against.
        """
        # Calculate the average for each metric
        average_metrics = {key: self.summed_metrics[key] / self.count_metrics[key] for key in self.summed_metrics}
        
        # Log the averages with wandb
        for key, avg_value in average_metrics.items():
            wandb.log({f"{self.title}/{key}": avg_value}, step=step)

        # Reset
        self.metrics = []
        self.summed_metrics = {}
        self.count_metrics = {}
        

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
    parser.add_argument('--reg_activation', type=str, default='sigmoid', help='Random seed.')  
    parser.add_argument('--vit_model', type=str, default='dinov2', help='Random seed.') 
    parser.add_argument('--score_reg_enabled', type=str, default='False', help='Random seed.')
    parser.add_argument('--mae_reg_enabled', type=str, default='False', help='Random seed.')        
    parser.add_argument('--refine_up_depth', type=int, default=2, help='Random seed.')   
    #parser.add_argument('--batch_size', type=int, default=32, help='Random seed.')
    parser.add_argument('--transformer_decoder_depth', type=int, default=1, help='Random seed.')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Random seed.')
    parser.add_argument('--refine_scale1', type=float, default=0.01, help='Random seed.')
    parser.add_argument('--refine_scale2', type=float, default=0.01, help='Random seed.')
    parser.add_argument('--refine_scale3', type=float, default=0.01, help='Random seed.')
    parser.add_argument('--refine_scale4', type=float, default=0.01, help='Random seed.')
    parser.add_argument('--score_reg_scale', type=float, default=0.000003, help='Random seed.')
    parser.add_argument('--aug_crop_scale', type=float, default=0.75, help='Random seed.')
    parser.add_argument('--aug_rot_deg', type=float, default=180.0, help='Random seed.')
    parser.add_argument('--dists_pref2ref_coeff', type=float, default=0.35, help='Random seed.')
    parser.add_argument('--l1_coeff', type=float, default=0.8, help='Random seed.')
    parser.add_argument('--lr', type=float, default=5e-3, help='Random seed.')

    # Parse arguments
    args = parser.parse_args()

    epochs = 4
    config = {
        "epochs": epochs,
        "loader_num_workers": 4,
        "beta1": 0.99,
        "beta2": 0.9999,
        "eps": 1e-8,
        "batch_size": DEVICE_BATCH_SIZE
    }     
    config.update(vars(args))

    exp_name=f"v7-l1-bs:{config['batch_size']}-lr:{config['lr']:.0e}-b1:{config['beta1']:.2f}-b2:{config['beta2']:.3f}"

    # Initialize wandb with the parsed arguments, further simplifying parameter names
    wandb.init(project='nerf-nr-qa', name=exp_name, config=config)
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATA_DIR = "/home/ccl/Datasets/NeRF-NR-QA/"  # Specify the path to your DATA_DIR

    train_metrics = MetricAggregator("Training Metrics Dict")
    val_metrics = MetricAggregator("Validation Metrics Dict")

    # CSV file 
    scores_df = pd.read_csv("/home/ccl/Datasets/NeRF-NR-QA/output.csv")
    val_scenes = ['scannerf_zebra', 'scannerf_cheetah', 'nerfstudio_stump', 'mipnerf360_garden']
    train_df = scores_df[~scores_df['scene'].isin(val_scenes)].reset_index() # + ['trex', 'horns']
    black_list_train_methods = [
        'instant-ngp-10', 'instant-ngp-20', 'instant-ngp-50', 'instant-ngp-100', 'instant-ngp-200',
        'nerfacto-10', 'nerfacto-20', 'nerfacto-50', 'nerfacto-100', 'nerfacto-200',
    ]
    train_df = train_df[~train_df['method'].isin(black_list_train_methods)].reset_index()

    val_df = scores_df[scores_df['scene'].isin(val_scenes)].reset_index()
    black_list_val_methods = [
        'instant-ngp-10', 'instant-ngp-20', 'instant-ngp-50', 'instant-ngp-100', 'instant-ngp-200', 
        'nerfacto-10', 'nerfacto-20', 'nerfacto-50', 'nerfacto-100', 'nerfacto-200', 
    ]
    val_df = val_df[~val_df['method'].isin(black_list_val_methods)].reset_index()

    test_df = pd.read_csv(TEST_SCORE_FILE)
    test_size = test_df.shape[0]

    train_dataset = NerfNRQADataset(train_df, dir = DATA_DIR, mode='gt', is_train=True, aug_crop_scale=config.aug_crop_scale, aug_rot_deg=config.aug_rot_deg)
    val_dataset = NerfNRQADataset(val_df, dir = DATA_DIR, mode='gt')
    
    train_sampler = SceneBalancedSampler(train_dataset)
    val_sampler = SceneBalancedSampler(val_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=recursive_collate, batch_size = DEVICE_BATCH_SIZE, num_workers=config.loader_num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, collate_fn=recursive_collate, batch_size = DEVICE_BATCH_SIZE, num_workers=config.loader_num_workers, pin_memory=True)

    

    model = NRModel(device=device, refine_up_depth=config.refine_up_depth)
    optimizer = optim.Adam(model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps
    )
    step = 0
    eval_step = 0
    test_step = 0
    update_step = 0
    for epoch in range(config.epochs):

        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("load_data"):

            model.train()
            for batch in tqdm(train_dataloader):
                gt_image, render, score_std, score_mean, render_id, frame_id = batch_to_device(batch, device)
                
                optimizer.zero_grad()
                with record_function("model_inference"):
                    losses = model.losses(gt_image, render, score_std, score_mean)
                loss = losses['combined']
                loss.backward()
                step += score_mean.shape[0]
                train_metrics.add_metric(losses)

                optimizer.step()
                train_metrics.log_summary(step)

            model.eval()
            for batch in tqdm(val_dataloader):
                gt_image, render, score_std, score_mean, render_id, frame_id = batch_to_device(batch, device)
                with torch.no_grad():
                    losses = model.losses(gt_image, render, score_std, score_mean)
                val_metrics.add_metric(losses)
            val_metrics.log_summary(step)
                     

            if (epoch+1) % 4 == 0:

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
                    test_scores_df = test_df.copy()
                    test_scores_df['TEST_SCORE'] = video_scores
                    tnt_files = ['truck_reference.mp4', 'playground_reference.mp4',
                    'train_reference.mp4', 'm60_reference.mp4']
                    syn_files = ['lego_reference.mp4', 'drums_reference.mp4',
                    'ficus_reference.mp4', 'ship_reference.mp4']
                    
                    syn_df = test_scores_df[test_scores_df['reference_filename'].isin(syn_files)].reset_index()
                    tnt_df = test_scores_df[test_scores_df['reference_filename'].isin(tnt_files)].reset_index()
                    video_scores = tnt_df['TEST_SCORE'].values
                    corr = compute_correlations(video_scores, tnt_df['MOS'].values)
                    corr['l1'] = np.mean(np.abs(video_scores - tnt_df['DISTS'].values))
                    for key in corr.keys():
                        metric = corr[key]
                        wandb.log({
                            f"Test Metrics Dict/tnt/mos/{key}": metric
                        }, step=step)

                    corr = compute_correlations(video_scores, tnt_df['DMOS'].values)
                    for key in corr.keys():
                        metric = corr[key]
                        wandb.log({
                            f"Test Metrics Dict/tnt/dmos/{key}": metric
                        }, step=step)

                    
                    video_scores = syn_df['TEST_SCORE'].values
                    corr = compute_correlations(video_scores, syn_df['MOS'].values)
                    corr['l1'] = np.mean(np.abs(video_scores - syn_df['DISTS'].values))
                    for key in corr.keys():
                        metric = corr[key]
                        wandb.log({
                            f"Test Metrics Dict/syn/mos/{key}": metric
                        }, step=step)

                    corr = compute_correlations(video_scores, syn_df['DMOS'].values)
                    for key in corr.keys():
                        metric = corr[key]
                        wandb.log({
                            f"Test Metrics Dict/syn/dmos/{key}": metric
                        }, step=step)


                    video_scores = test_scores_df['TEST_SCORE'].values
                    corr = compute_correlations(video_scores, test_scores_df['MOS'].values)
                    corr['l1'] = np.mean(np.abs(video_scores - test_scores_df['DISTS'].values))
                    for key in corr.keys():
                        metric = corr[key]
                        wandb.log({
                            f"Test Metrics Dict/mos/{key}": metric
                        }, step=step)

                    corr = compute_correlations(video_scores, test_scores_df['DMOS'].values)
                    for key in corr.keys():
                        metric = corr[key]
                        wandb.log({
                            f"Test Metrics Dict/dmos/{key}": metric
                        }, step=step)

    
    model.train()

    # %%
