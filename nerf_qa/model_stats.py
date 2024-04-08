#%%
# deep learning
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

import math

# local
from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def linear_func(x, a, b):
    return a * x + b


class NeRFQAModel(nn.Module):
    def __init__(self, train_df, mode = 'mean'):
        super(NeRFQAModel, self).__init__()

        self.mode = mode
        if self.mode == 'mean-std-min-max':
            X = np.transpose(np.stack([
                train_df['DISTS'].values,
                train_df['DISTS_std'].values,
                train_df['DISTS_min'].values,
                train_df['DISTS_max'].values,
            ]))
        elif self.mode == 'mean-std':
            X = np.transpose(np.stack([
                train_df['DISTS'].values,
                train_df['DISTS_std'].values,
            ]))
        else:
            X = np.transpose(np.array([
                train_df['DISTS'].values,
            ]))
        
        print("X.shape", X.shape)
        y = train_df['MOS'].values  # Response

        # Create a linear regression model to initialize linear layer
        model = LinearRegression()
        model.fit(X, y)

        # Print the coefficients
        print(f"Coefficient: {model.coef_}")
        print(f"Intercept: {model.intercept_}")
        self.dists_model = DISTS()
        self.dists_weight = nn.Parameter(torch.tensor([model.coef_], dtype=torch.float32).T)
        self.dists_bias = nn.Parameter(torch.tensor([model.intercept_], dtype=torch.float32))
            
    
    def compute_dists_with_batches(self, dataloader):
        all_scores = []  # Collect scores from all batches as tensors

        for dist_batch, ref_batch in dataloader:
            ref_images = ref_batch.to(device)  # Assuming ref_batch[0] is the tensor of images
            dist_images = dist_batch.to(device)  # Assuming dist_batch[0] is the tensor of images
            with torch.no_grad():
                feats0 = self.dists_model.forward_once(dist_images)
                feats1 = self.dists_model.forward_once(ref_images) 
                scores = self.dists_model.forward_from_feats(feats0, feats1)
                # scores = self.forward(ref_images, dist_images)  # Returns a tensor of scores
            
            # Collect scores tensors
            all_scores.append(scores)

        # Concatenate all score tensors into a single tensor
        all_scores_tensor = torch.cat(all_scores, dim=0)
        if all_scores_tensor.numel() > 0:
            score_mean = torch.mean(all_scores_tensor, dim=0, keepdim=True)
            score_std = torch.mean(all_scores_tensor, dim=0, keepdim=True)
            score_min, _ = torch.min(all_scores_tensor, dim=0, keepdim=True)
            score_max, _ = torch.max(all_scores_tensor, dim=0, keepdim=True)
        else:
            score_mean = all_scores_tensor
            score_std = torch.zeros_like(score_mean)
            score_min = score_mean
            score_max = score_mean
        
        if self.mode == 'mean-std-min-max':
            agg_score = torch.stack([score_mean, score_std, score_min, score_max], dim=1)
        elif self.mode == 'mean-std':
            agg_score = torch.stack([score_mean, score_std], dim=1)
        else:
            agg_score = score_mean.unsqueeze(1)
        final_score = (agg_score @ self.dists_weight).squeeze(1) + self.dists_bias
        return final_score.squeeze()
        
    def forward(self, dist, ref, stats = None):
        with torch.no_grad():
            feats0 = self.dists_model.forward_once(dist)
            feats1 = self.dists_model.forward_once(ref) 
        dists_scores = self.dists_model.forward_from_feats(feats0, feats1)
        

        if self.mode == 'mean-std-min-max' or self.mode == 'mean-std':
            dists_scores = torch.concat([
                dists_scores.unsqueeze(1),
                stats
            ], dim=1)
        else:
            dists_scores = dists_scores.unsqueeze(1)
        scores = (dists_scores @ self.dists_weight).squeeze(1) + self.dists_bias # linear function
        return scores

