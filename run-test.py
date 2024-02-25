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
from scipy.optimize import curve_fit

# data 
import pandas as pd
import cv2
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go

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
    def __init__(self, train_df = None, linearization_type = 'linear'):
        super(VQAModel, self).__init__()
        self.linearization_type = linearization_type

        # Print the coefficients
        self.dists_model = DISTS()
        if train_df:
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
            print(f"Coefficient: {model.coef_[0]}")
            print(f"Intercept: {model.intercept_}")

            self.dists_weight = nn.Parameter(torch.tensor([model.coef_[0]], dtype=torch.float32))
            self.dists_bias = nn.Parameter(torch.tensor([model.intercept_], dtype=torch.float32))
        else:
            self.dists_weight = nn.Parameter(torch.tensor([1], dtype=torch.float32))
            self.dists_bias = nn.Parameter(torch.tensor([0], dtype=torch.float32))


    def compute_dists_with_batches(self, dataloader):
        all_scores = []  # Collect scores from all batches as tensors

        for dist_batch, ref_batch in dataloader:
            ref_images = ref_batch.to(device)  # Assuming ref_batch[0] is the tensor of images
            dist_images = dist_batch.to(device)  # Assuming dist_batch[0] is the tensor of images
            scores = self.dists_model(ref_images, dist_images, require_grad=False, batch_average=False)  # Returns a tensor of scores
            
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


val_df = scores_df
val_size = val_df.shape[0]

print(f"Validation Refrences: {val_df['reference_filename'].drop_duplicates().values}")

# Reset model and optimizer for each fold (if you want to start fresh for each fold)
model = VQAModel().to(device)
model.load_state_dict(torch.load('ckpt/checkpoint_4_l1_1645.pt'))

print(model.dists_weight, model.dists_bias)

mse_fn = nn.MSELoss(reduction='none')

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
            loss = mse_fn(predicted_score, target_score)
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



        # Log to wandb
        wandb.log({
            f"Test Metrics Dict/batch_loss": eval_loss,
            f"Test Metrics Dict/rmse": rsme,
            f"Test Metrics Dict/plcc": plcc,
            f"Test Metrics Dict/srcc": srcc,
        }, step=0)
        wandb.log({
            f"Test Metrics Dict/rmse_hist": wandb.Histogram(np.array(all_rmse)),
        }, step=0)
        dists_mos_log_fig = plot_dists_mos_log(val_df)
        dists_ft_mos_lin_fig = plot_dists_ft_mos(all_target_scores, all_predicted_scores)
        wandb.log({
            f"Test Plots/dists_mos_log": wandb.Plotly(dists_mos_log_fig),
            f"Test Plots/dists_ft_mos_lin_fig": wandb.Plotly(dists_ft_mos_lin_fig)
        }, step=0)

    