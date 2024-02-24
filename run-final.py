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
from nerf_qa.settings import DEVICE_BATCH_SIZE

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
parser.add_argument('--linearization_type', type=lambda x: (str(x).lower() in ['linear', 'log', 'sqrt']), default='linear', help='Whether to resize images.')
parser.add_argument('--batch_size', type=int, default=1645, help='Batch size.')

# Further simplified optimizer configurations
parser.add_argument('--lr_scale', type=float, default=1.0, help='Learning rate.')
parser.add_argument('--eps_scale', type=float, default=5.0, help='Optimizer epsilon.')
parser.add_argument('--beta1_scale', type=float, default=0.92, help='Optimizer beta1.')
parser.add_argument('--beta2_scale', type=float, default=1.0, help='Optimizer beta2.')
parser.add_argument('--delta', type=float, default=0.075, help='Huber loss delta')
parser.add_argument('--gamma', type=float, default=0.995, help='Huber loss delta')
parser.add_argument('--kappa_scale', type=float, default=1.0, help='Kappa scale')


# Parse arguments
args = parser.parse_args()

kappa = args.kappa_scale * float(args.batch_size) / float(1024)
batches_per_step = -(args.batch_size // -DEVICE_BATCH_SIZE)
epochs = 256
config = {
    "epochs": epochs,
    "batches_per_step": batches_per_step,
    "kappa": kappa,
    "lr": 5e-5,
    "beta1": 0.4,
    "beta2": 0.95,
    "eps": 1e-9,
}     
config.update(vars(args))


#%%

# Initialize wandb with the parsed arguments, further simplifying parameter names
wandb.init(project='nerf-qa', config=config)
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
    
class EarlyStoppingWithMA:
    def __init__(self, patience=10, verbose=False, delta=0.03, path='checkpoint_4_l1_1645.pt', trace_func=print, ma_window=10):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.ma_window = ma_window  # Window size for the moving average
        self.scores = []  # Keep track of validation losses to calculate moving average

    def __call__(self, score, model):
        self.scores.append(score)
        # Calculate the moving average of the validation losses
        if len(self.scores) > self.ma_window:
            ma_score = np.mean(self.scores[-self.ma_window:])
        else:
            ma_score = np.mean(self.scores)
        
        # The rest of the logic remains similar, but compare the moving average
        score = ma_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)

#%%
# Read the CSV file
scores_df = pd.read_csv(SCORE_FILE)
# filter test
test_scenes = ['ship', 'lego', 'drums', 'ficus', 'train', 'm60', 'playground', 'truck']
train_df = scores_df[~scores_df['scene'].isin(test_scenes)].reset_index()
test_df = scores_df[scores_df['scene'].isin(test_scenes)].reset_index()

mse_fn = nn.MSELoss(reduction='none')
loss_fn = nn.L1Loss(reduction='none')


#%%

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
def create_dataloader(scores_df):
    # Create a dataset and dataloader for efficient batching
    dataset = LargeQADataset(dir=DATA_DIR, scores_df=scores_df)
    dataloader = DataLoader(dataset, batch_size=DEVICE_BATCH_SIZE, shuffle=True)
    return dataloader


global_step = 0
weighted_score_est = 0
plccs = []
srccs = []
rmses = []



print(f"Validation Refrences: {test_df['scene'].drop_duplicates().values}")

# Reset model and optimizer for each fold (if you want to start fresh for each fold)
model = VQAModel(train_df=train_df, linearization_type=config.linearization_type).to(device)
optimizer = optim.Adam(model.parameters(),
    lr=config.lr,
    betas=(config.beta1, config.beta2),
    eps=config.eps
)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
early_stopper = EarlyStoppingWithMA()
step_early_stop = 0
plcc_early_stop = 0
srcc_early_stop = 0
rmse_early_stop = float("inf")
weighted_score_early_stop = 0

train_dataloader = create_dataloader(train_df)
test_dataloader = create_dataloader(test_df)
train_size = len(train_dataloader)
test_size = len(test_dataloader)
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
        mse = torch.dot(mse, weights)
        # Accumulate gradients
        loss.backward()
        total_loss += loss.item()
        batch_loss += loss.item()
        batch_mse += mse.item()
        
        if index % config.batches_per_step == 0 or index == train_size:

            # Scale gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad /= weight_sum
            
            # Update parameters every batches_per_step steps or on the last iteration
            optimizer.step()
            optimizer.zero_grad()  # Zero the gradients after updating
            average_batch_loss = batch_loss / weight_sum
            wandb.log({
                f"Train Metrics Dict/batch_loss": average_batch_loss,
                f"Train Metrics Dict/rmse": np.sqrt(batch_mse / weight_sum),
                }, step=global_step)
            batch_loss = 0
            batch_mse = 0
            weight_sum = 0
    
    # Validation step
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        batch_loss = 0
        all_rmse = []
        all_target_scores = []  # List to store all target scores
        all_predicted_scores = []  # List to store all predicted scores
        all_ids = []  # List to store all predicted scores
        
        for dist, ref, score, i in tqdm(test_dataloader, total=test_size, desc="Validating..."):
            # Compute score
            predicted_score = model(dist.to(device), ref.to(device))
            target_score = score.to(device).float()
            all_predicted_scores.append(predicted_score.cpu())
            all_target_scores.append(target_score.cpu())
        
            # Compute loss
            loss = loss_fn(predicted_score, target_score).mean()
            mse = mse_fn(predicted_score, target_score).mean()
            batch_loss += loss.item()
            all_rmse.append(float(np.sqrt(mse.item())))
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
        batch_loss /= len(test_df)
        rmse = np.mean(all_rmse)
        weighted_score = 1.0*float(plcc) + 1.0*float(srcc) - 0.5*float(rmse)

        if weighted_score > weighted_score_early_stop:
            step_early_stop = global_step
            plcc_early_stop = float(plcc)
            srcc_early_stop = float(srcc)
            rmse_early_stop = float(rmse)
            weighted_score_early_stop = weighted_score

        early_stopper(weighted_score, model)

        if epoch == wandb.config.epochs-1 or early_stopper.early_stop:
            # last epoch
            plccs.append(plcc_early_stop)
            srccs.append(srcc_early_stop)
            rmses.append(rmse_early_stop)

        # Log to wandb
        wandb.log({
            f"Test Metrics Dict/batch_loss": batch_loss,
            f"Test Metrics Dict/rmse": rmse,
            f"Test Metrics Dict/plcc": plcc,
            f"Test Metrics Dict/srcc": srcc,
            f"Test Metrics Dict/weighted_score": weighted_score,
        }, step=global_step)
        
        dists_mos_log_fig = plot_dists_mos_log(test_df)
        dists_ft_mos_lin_fig = plot_dists_ft_mos(all_target_scores, all_predicted_scores)
        wandb.log({
            f"Test Plots/dists_mos_log": wandb.Plotly(dists_mos_log_fig),
            f"Test Plots/dists_ft_mos_lin_fig": wandb.Plotly(dists_ft_mos_lin_fig)
        }, step=global_step)

        scheduler.step()
        if early_stopper.early_stop:
            break


        
    # Logging the average loss
    average_loss = total_loss / len(train_df)
    print(f"Average Loss: {average_loss}, Est Weighted Score: {weighted_score_est}\n\n")


#%%


#%%

# %%
wandb.finish()
# %%
