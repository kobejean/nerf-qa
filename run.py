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

# data 
import pandas as pd
import cv2
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# local
from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS, prepare_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%%
DATA_DIR = "/home/ccl/Datasets/NeRF-QA"
REF_DIR = path.join(DATA_DIR, "Reference")
SYN_DIR = path.join(DATA_DIR, "NeRF-QA_videos")
SCORE_FILE = path.join(DATA_DIR, "NeRF_VQA_MOS.csv")

import argparse
import wandb

# Set up argument parser
parser = argparse.ArgumentParser(description='Initialize a new run with wandb with custom configurations.')

# Basic configurations
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--resize', type=lambda x: (str(x).lower() in ['true', '1', 'yes', 'y']), default=True, help='Whether to resize images.')
parser.add_argument('--epochs', type=int, default=80, help='Number of epochs.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
parser.add_argument('--frame_batch_size', type=int, default=32, help='Frame batch size, affects training time and memory usage.')

# Further simplified optimizer configurations
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--eps', type=float, default=1e-8, help='Optimizer epsilon.')
parser.add_argument('--beta1', type=float, default=0.9, help='Optimizer beta1.')
parser.add_argument('--beta2', type=float, default=0.999, help='Optimizer beta2.')

# Parse arguments
args = parser.parse_args()

# Initialize wandb with the parsed arguments, further simplifying parameter names
wandb.init(project='nerf-qa', config=args)

# Access the config
config = wandb.config


#%%

class VQAModel(nn.Module):
    def __init__(self, train_df):
        super(VQAModel, self).__init__()
        # Reshape data (scikit-learn expects X to be a 2D array)
        X = train_df['DISTS'].values.reshape(-1, 1)  # Predictor
        y = train_df['MOS'].values  # Response

        # Create a linear regression model
        model = LinearRegression()

        # Fit the model
        model.fit(X, y)

        # Print the coefficients
        print(f"Coefficient: {model.coef_[0]}")
        print(f"Intercept: {model.intercept_}")
        self.dists_model = DISTS()
        self.dists_weight = nn.Parameter(torch.tensor([model.coef_[0]], dtype=torch.float32))
        self.dists_bias = nn.Parameter(torch.tensor([model.intercept_], dtype=torch.float32))

    def compute_dists_with_batches(self, dataloader):
        all_scores = []  # Collect scores from all batches as tensors

        for dist_batch, ref_batch in dataloader:
            ref_images = ref_batch.to(device)  # Assuming ref_batch[0] is the tensor of images
            dist_images = dist_batch.to(device)  # Assuming dist_batch[0] is the tensor of images
            scores = self.dists_model(ref_images, dist_images, require_grad=True, batch_average=False)  # Returns a tensor of scores
            
            # Collect scores tensors
            all_scores.append(scores)

        # Concatenate all score tensors into a single tensor
        all_scores_tensor = torch.cat(all_scores, dim=0)

        # Compute the average score across all batches
        average_score = torch.mean(all_scores_tensor) if all_scores_tensor.numel() > 0 else torch.tensor(0.0).to(device)

        return average_score
        
    def forward(self, dataloader):
        raw_scores = self.compute_dists_with_batches(dataloader)
        
        # Normalize raw scores using the trainable mean and std
        normalized_scores = raw_scores * self.dists_weight + self.dists_bias
        return normalized_scores


#%%
# Read the CSV file
scores_df = pd.read_csv(SCORE_FILE)
# filter test
test_files = ['ship_reference.mp4', 'truck_reference.mp4']
scores_df = scores_df[~scores_df['reference_filename'].isin(test_files)]

loss_fn = nn.MSELoss()



# Number of splits for GroupKFold
num_folds = min(scores_df['reference_filename'].nunique(), 3)

# Example function to load a video and process it frame by frame
def load_video_frames(video_path):
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
        frame = prepare_image(frame, resize=config.resize).squeeze(0)
        frames.append(frame)
    cap.release()
    return torch.stack(frames)

# Batch creation function
def create_dataloader(row, frame_batch_size):
    dist_video_path = path.join(SYN_DIR, row['distorted_filename'])
    ref_video_path = path.join(REF_DIR, row['reference_filename'])
    ref = load_video_frames(ref_video_path)
    dist = load_video_frames(dist_video_path)
    # Create a dataset and dataloader for efficient batching
    dataset = TensorDataset(dist, ref)
    dataloader = DataLoader(dataset, batch_size=frame_batch_size, shuffle=False)
    return dataloader

# Initialize GroupKFold
gkf = GroupKFold(n_splits=num_folds)

# Extract reference filenames as groups for GroupKFold
groups = scores_df['reference_filename'].values

global_step = 0
plccs = []
srccs = []
rsmes = []

# Group K-Fold Cross-Validation
for fold, (train_idx, val_idx) in enumerate(gkf.split(scores_df, groups=groups), 1):
    print(f"Fold {fold}/{num_folds}")
    
    # Split the data into training and validation sets
    train_df = scores_df.iloc[train_idx]
    val_df = scores_df.iloc[val_idx]
    train_size = train_df.shape[0]
    val_size = val_df.shape[0]

    print(f"Validation Refrences: {val_df['reference_filename'].drop_duplicates().values}")

    # Reset model and optimizer for each fold (if you want to start fresh for each fold)
    model = VQAModel(train_df=train_df).to(device)
    optimizer = optim.Adam(model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps
    )

    step_early_stop = 0
    plcc_early_stop = 0
    srcc_early_stop = 0
    rsme_early_stop = float("inf")

    # Training loop
    for epoch in range(wandb.config.epochs):
        print(f"Epoch {epoch+1}/{wandb.config.epochs}")
        model.train()  # Set model to training mode
        total_loss = 0
        batch_loss = 0
        optimizer.zero_grad()  # Initialize gradients to zero at the start of each epoch

        # Shuffle train_df with random seed
        train_df = train_df.sample(frac=1, random_state=config.seed+global_step).reset_index(drop=True)
        for index, (i, row) in tqdm(enumerate(train_df.iterrows(), 1), total=train_size, desc="Training..."):  # Start index from 1 for easier modulus operation
            # Load frames
            dataloader = create_dataloader(row, config.frame_batch_size)
            
            # Compute score
            predicted_score = model(dataloader)
            target_score = torch.tensor(row['MOS'], device=device, dtype=torch.float32)
            
            # Compute loss
            loss = loss_fn(predicted_score, target_score)
            
            # Accumulate gradients
            loss.backward()
            total_loss += loss.item()
            batch_loss += loss.item()
            
            if index % config.batch_size == 0 or index == train_size:

                # Scale gradients
                accumulation_steps = ((index-1) % config.batch_size) + 1
                global_step += accumulation_steps
                if accumulation_steps > 1:
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad /= accumulation_steps
                
                # Update parameters every batch_size steps or on the last iteration
                optimizer.step()
                optimizer.zero_grad()  # Zero the gradients after updating
                average_batch_loss = batch_loss / config.batch_size
                wandb.log({
                    f"Train Metrics Dict/batch_loss/k{fold}": average_batch_loss,
                    f"Train Metrics Dict/rmse/k{fold}": np.sqrt(average_batch_loss),
                    }, step=global_step)
                batch_loss = 0
        
        # Validation step
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            eval_loss = 0
            all_rmse = []
            all_target_scores = []  # List to store all target scores
            all_predicted_scores = []  # List to store all predicted scores

            for index, row in tqdm(val_df.iterrows(), total=val_size, desc="Validating..."):
                # Load frames
                dataloader = create_dataloader(row, config.frame_batch_size)
                
                # Compute score
                predicted_score = model(dataloader)
                target_score = torch.tensor(row['MOS'], device=device, dtype=torch.float32)
                all_predicted_scores.append(float(predicted_score.item()))
                all_target_scores.append(float(target_score.item()))
            
                # Compute loss
                loss = loss_fn(predicted_score, target_score)
                eval_loss += loss.item()
                all_rmse.append(float(np.sqrt(loss.item())))

            
            # Convert lists to arrays for correlation computation
            all_target_scores = np.array(all_target_scores)
            all_predicted_scores = np.array(all_predicted_scores)
            
            # Compute PLCC and SRCC
            plcc = pearsonr(all_target_scores, all_predicted_scores)[0]
            srcc = spearmanr(all_target_scores, all_predicted_scores)[0]
            
            # Average loss over validation set
            eval_loss /= len(val_df)
            rsme = np.mean(all_rmse)

            if rsme < rsme_early_stop:
                step_early_stop = global_step
                plcc_early_stop = float(plcc)
                srcc_early_stop = float(srcc)
                rsme_early_stop = float(rsme)

            if epoch == wandb.config.epochs-1:
                # last epoch
                plccs.append(plcc_early_stop)
                srccs.append(srcc_early_stop)
                rsmes.append(rsme_early_stop)

            # Log to wandb
            wandb.log({
                f"Eval Metrics Dict/batch_loss/k{fold}": eval_loss,
                f"Eval Metrics Dict/rmse/k{fold}": rsme,
                f"Eval Metrics Dict/plcc/k{fold}": plcc,
                f"Eval Metrics Dict/srcc/k{fold}": srcc,
            }, step=global_step)
            wandb.log({
                f"Eval Metrics Dict/rmse_hist/k{fold}": wandb.Histogram(np.array(all_rmse)),
            }, step=global_step)

            
        # Logging the average loss
        average_loss = total_loss / len(scores_df)
        print(f"Average Loss: {average_loss}\n\n")
        wandb.log({ f"Train Metrics Dict/total_loss/k{fold}": average_batch_loss }, step=global_step)

weighted_score = -1.0 * np.mean(rsmes) + 1.0 * np.mean(plccs) + 1.0 * np.mean(srccs)
# Log to wandb
wandb.log({
    "Cross-Validation Metrics Dict/weighted_score_mean": weighted_score,
    "Cross-Validation Metrics Dict/rmse_mean": np.mean(rsmes),
    "Cross-Validation Metrics Dict/rmse_std": np.std(rsmes),
    "Cross-Validation Metrics Dict/plcc_mean": np.mean(plccs),
    "Cross-Validation Metrics Dict/plcc_std": np.std(plccs),
    "Cross-Validation Metrics Dict/srcc_mean": np.mean(srccs),
    "Cross-Validation Metrics Dict/srcc_std": np.std(srccs),
}, step=global_step)
wandb.log({
    "Cross-Validation Metrics Dict/rmse_hist": wandb.Histogram(np.array(rsmes)),
    "Cross-Validation Metrics Dict/plcc_hist": wandb.Histogram(np.array(plccs)),
    "Cross-Validation Metrics Dict/srcc_hist": wandb.Histogram(np.array(srccs)),
}, step=global_step)

#%%