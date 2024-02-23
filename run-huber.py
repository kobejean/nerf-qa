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
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# local
from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS, prepare_image
from nerf_qa.data import LargeQADataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%%
DATA_DIR = "/home/ccl/Datasets/NeRF-QA-Large-1"
SCORE_FILE = path.join(DATA_DIR, "scores.csv")

import argparse
import wandb

# Set up argument parser
parser = argparse.ArgumentParser(description='Initialize a new run with wandb with custom configurations.')

# Basic configurations
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--resize', type=lambda x: (str(x).lower() in ['true', '1', 'yes', 'y']), default=True, help='Whether to resize images.')
parser.add_argument('--linearization_type', type=lambda x: (str(x).lower() in ['linear', 'log', 'sqrt']), default='sqrt', help='Whether to resize images.')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs.')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size.')
parser.add_argument('--frame_batch_size', type=int, default=32, help='Frame batch size, affects training time and memory usage.')

# Further simplified optimizer configurations
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--eps', type=float, default=1e-8, help='Optimizer epsilon.')
parser.add_argument('--beta1', type=float, default=0.9, help='Optimizer beta1.')
parser.add_argument('--beta2', type=float, default=0.999, help='Optimizer beta2.')
parser.add_argument('--delta', type=float, default=1.0, help='Huber loss delta')

# Parse arguments
args = parser.parse_args()
# Simulate args
class Args:
    def __init__(self):
        self.linearization_type = 'sqrt'
        self.seed = 42
        self.resize = True  # Modify this to False if you don't want resizing.
        self.epochs = 8
        self.batch_size = 8
        self.frame_batch_size = 32
        self.lr = 3e-5
        self.eps = 1e-8
        self.beta1 = 0.9
        self.beta2 = 0.999

# Instead of parsing args, create an instance of the Args class
#args = Args()
#%%

# Initialize wandb with the parsed arguments, further simplifying parameter names
wandb.init(project='nerf-qa', config=args)

# Access the config
config = wandb.config


#%%

class VQAModel(nn.Module):
    def __init__(self, train_df, linearization_type = True):
        super(VQAModel, self).__init__()
        self.linearization_type = linearization_type
        # Reshape data (scikit-learn expects X to be a 2D array)
        X = train_df['DISTS'].values.reshape(-1, 1)  # Predictor
        if self.linearization_type == 'log':
            X = np.log(X)
        elif self.linearization_type == 'sqrt':
            X = np.sqrt(X)
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

        
    def forward(self, dist, ref):
        scores = self.dists_model(ref, dist, require_grad=True, batch_average=False)  # Returns a tensor of scores
        
        if self.linearization_type == 'log':
            scores = torch.log(scores)
        elif self.linearization_type == 'sqrt':
            scores = torch.sqrt(scores)
        # Normalize raw scores using the trainable mean and std
        normalized_scores = scores * self.dists_weight + self.dists_bias
        return normalized_scores


#%%
# Read the CSV file
scores_df = pd.read_csv(SCORE_FILE)
# filter test
test_scenes = ['ship', 'lego', 'drums', 'ficus', 'train', 'm60', 'playground', 'truck']
scores_df = scores_df[~scores_df['scene'].isin(test_scenes)]

mse_fn = nn.MSELoss(reduction='none')
loss_fn = nn.HuberLoss(reduction='none', delta=config.delta)


#%%
# Number of splits for GroupKFold
num_folds = min(scores_df['scene'].nunique(), 2)

#
def plot_dists_mos_log(df):
    x_data = df['MOS']
    y_data = df['DISTS']
    def log_func(x, a, b):
        return a + b * np.log(x)
    # Fit the model
    params, params_covariance = curve_fit(log_func, x_data, y_data)

    # Predict using the fitted model
    x_range = np.linspace(min(x_data), max(x_data), 400)
    y_pred = log_func(x_range, params[0], params[1])

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='Data'))

    # Regression line
    fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='Logarithmic Regression'))
    fig.update_layout(title='Logarithmic Regression between DISTS and MOS',
                    xaxis_title='MOS',
                    yaxis_title='DISTS')
    return fig

def plot_dists_ft_mos(all_target_scores, all_predicted_scores):
    x_data = all_target_scores
    y_data = all_predicted_scores
    def lin_func(x, a, b):
        return a + b * x
    # Fit the model
    params, params_covariance = curve_fit(lin_func, x_data, y_data)

    # Predict using the fitted model
    x_range = np.linspace(min(x_data), max(x_data), 400)
    y_pred = lin_func(x_range, params[0], params[1])
    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='Data'))

    # Regression line
    fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name='Linear Regression'))
    fig.update_layout(title='Linear Regression between DISTS Fine-tuned and MOS',
                    xaxis_title='MOS',
                    yaxis_title='DISTS (Fine-tuned)')
    return fig


# Batch creation function
def create_dataloader(scores_df, frame_batch_size):
    # Create a dataset and dataloader for efficient batching
    dataset = LargeQADataset(dir=DATA_DIR, scores_df=scores_df)
    dataloader = DataLoader(dataset, batch_size=frame_batch_size, shuffle=True)
    return dataloader

# Initialize GroupKFold
gkf = GroupKFold(n_splits=num_folds)

# Extract reference filenames as groups for GroupKFold
groups = scores_df['scene'].values

global_step = 0
plccs = []
srccs = []
rsmes = []

# Group K-Fold Cross-Validation
for fold, (train_idx, val_idx) in enumerate(gkf.split(scores_df, groups=groups), 1):
    print(f"Fold {fold}/{num_folds}")
    
    # Split the data into training and validation sets
    train_df = scores_df.iloc[train_idx].reset_index()
    val_df = scores_df.iloc[val_idx].reset_index()

    print(f"Validation Refrences: {val_df['scene'].drop_duplicates().values}")

    # Reset model and optimizer for each fold (if you want to start fresh for each fold)
    model = VQAModel(train_df=train_df, linearization_type=config.linearization_type).to(device)
    optimizer = optim.Adam(model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps
    )

    step_early_stop = 0
    plcc_early_stop = 0
    srcc_early_stop = 0
    rsme_early_stop = float("inf")

    train_dataloader = create_dataloader(train_df, config.frame_batch_size)
    val_dataloader = create_dataloader(val_df, config.frame_batch_size)
    train_size = len(train_dataloader)
    val_size = len(val_dataloader)
    # Training loop
    for epoch in range(wandb.config.epochs):
        print(f"Epoch {epoch+1}/{wandb.config.epochs}")
        model.train()  # Set model to training mode
        total_loss = 0
        batch_loss = 0
        batch_mse = 0
        weight_sum = 0
        optimizer.zero_grad()  # Initialize gradients to zero at the start of each epoch

        for index, (dist,ref,score,i) in tqdm(enumerate(train_dataloader, 1), total=train_size, desc="Training..."):  # Start index from 1 for easier modulus operation            
            # Compute score
            predicted_score = model(dist.to(device),ref.to(device))
            target_score = score.to(device).float()
            
            # Compute loss
            mse = mse_fn(predicted_score, target_score)
            loss = loss_fn(predicted_score, target_score)
            weights = 1 / torch.tensor(train_df['frame_count'].iloc[i.numpy()].values, device=device, dtype=torch.float32)

            global_step += weights.shape[0]
            weight_sum += weights.sum().item()
            loss = torch.dot(loss, weights)
            mse = torch.dot(loss, mse)
            # Accumulate gradients
            loss.backward()
            total_loss += loss.item()
            batch_loss += loss.item()
            batch_mse += mse.item()
            
            if index % config.batch_size == 0 or index == train_size:

                # Scale gradients
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= weight_sum
                
                # Update parameters every batch_size steps or on the last iteration
                optimizer.step()
                optimizer.zero_grad()  # Zero the gradients after updating
                average_batch_loss = batch_loss / weight_sum
                wandb.log({
                    f"Train Metrics Dict/batch_loss/k{fold}": average_batch_loss,
                    f"Train Metrics Dict/rmse/k{fold}": np.sqrt(batch_mse / weight_sum),
                    }, step=global_step)
                batch_loss = 0
                batch_mse = 0
                weight_sum = 0
        
        # Validation step
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            eval_loss = 0
            all_rmse = []
            all_target_scores = []  # List to store all target scores
            all_predicted_scores = []  # List to store all predicted scores
            all_ids = []  # List to store all predicted scores
            
            for dist, ref, score, i in tqdm(val_dataloader, total=val_size, desc="Validating..."):
                # Compute score
                predicted_score = model(dist.to(device), ref.to(device))
                target_score = score.to(device).float()
                all_predicted_scores.append(predicted_score.cpu())
                all_target_scores.append(target_score.cpu())
            
                # Compute loss
                loss = loss_fn(predicted_score, target_score).mean()
                mse = mse_fn(predicted_score, target_score).mean()
                eval_loss += loss.item()
                all_rmse.append(float(np.sqrt(lomsess.item())))
                all_ids.append(i.cpu())

            
            # Convert lists to arrays for correlation computation
            all_target_scores = np.concatenate(all_target_scores, axis=0)
            all_predicted_scores = np.concatenate(all_predicted_scores, axis=0)
            #all_rmse = np.concatenate(all_rmse, axis=0)
            all_ids = np.concatenate(all_ids, axis=0)

            

            # Step 1: Create a DataFrame
            df = pd.DataFrame({
                'ID': all_ids,
                'TargetScore': all_target_scores,
                'PredictedScore': all_predicted_scores,
            })

            # Step 2: Group by ID and calculate mean
            average_scores = df.groupby('ID').mean().reset_index()
            all_target_scores = average_scores['TargetScore'].values
            all_predicted_scores = average_scores['PredictedScore'].values
            #all_rmse = average_scores['RMSE'].values

            # Compute PLCC and SRCC
            plcc = pearsonr(all_target_scores, all_predicted_scores)[0]
            srcc = spearmanr(all_target_scores, all_predicted_scores)[0]
            all_target_scores
            
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
                # plccs.append(plcc_early_stop)
                # srccs.append(srcc_early_stop)
                # rsmes.append(rsme_early_stop)
                plccs.append(float(plcc))
                srccs.append(float(srcc))
                rsmes.append(float(rsme))

            # Log to wandb
            wandb.log({
                f"Eval Metrics Dict/batch_loss/k{fold}": eval_loss,
                f"Eval Metrics Dict/rmse/k{fold}": rsme,
                f"Eval Metrics Dict/plcc/k{fold}": plcc,
                f"Eval Metrics Dict/srcc/k{fold}": srcc,
            }, step=global_step)
            # wandb.log({
            #     f"Eval Metrics Dict/rmse_hist/k{fold}": wandb.Histogram(np.array(all_rmse)),
            # }, step=global_step)
            dists_mos_log_fig = plot_dists_mos_log(val_df)
            dists_ft_mos_lin_fig = plot_dists_ft_mos(all_target_scores, all_predicted_scores)
            wandb.log({
                f"Eval Plots/dists_mos_log/k{fold}": wandb.Plotly(dists_mos_log_fig),
                f"Eval Plots/dists_ft_mos_lin_fig/k{fold}": wandb.Plotly(dists_ft_mos_lin_fig)
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
# wandb.log({
#     "Cross-Validation Metrics Dict/rmse_hist": wandb.Histogram(np.array(rsmes)),
#     "Cross-Validation Metrics Dict/plcc_hist": wandb.Histogram(np.array(plccs)),
#     "Cross-Validation Metrics Dict/srcc_hist": wandb.Histogram(np.array(srccs)),
# }, step=global_step)

#%%


#%%

# %%
wandb.finish()
# %%
