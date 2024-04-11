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

        X = np.sqrt(np.transpose(np.array([
            train_df['DISTS'].values,
        ])))
    
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
            
  
    def forward(self, dist, ref, stats = None):
        dists_scores = self.dists_model(dist, ref)
        dists_scores = dists_scores.unsqueeze(1)
        scores = (torch.sqrt(dists_scores) @ self.dists_weight).squeeze(1) + self.dists_bias # linear function
        return scores, dists_scores

