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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def linear_func(x, a, b):
#     return a * x + b


class NeRFQAModel(nn.Module):
    def __init__(self, train_df, mode = 'normal'):
        super(NeRFQAModel, self).__init__()

        self.mode = mode

        X = train_df['DISTS'].values

        def logistic(x, beta1, beta2, beta3, beta4):
            return 2.0*(beta1 - beta2) / (1 + np.exp((x) / np.abs(beta4))) + beta2
    
        print("X.shape", X.shape)
        y = train_df['MOS'].values  # Response

        # Initial parameter guesses
        beta1_init = 5.0 #np.max(y)
        beta2_init = 1.0 #np.min(y)
        beta3_init = 0.0 #np.mean(X)
        beta4_init = np.std(X) / 4

        # Create a linear regression model to initialize linear layer
        # model = LinearRegression()
        # model.fit(X, y)

        params, params_covariance = curve_fit(logistic, X, y, p0=[beta1_init, beta2_init, beta3_init, beta4_init])

        # Print the coefficients
        print(f"Params: {params}")
        # print(f"Intercept: {model.intercept_}")  
        if wandb.config.mode in ["softmax"]:
            from nerf_qa.DISTS_pytorch.DISTS_pt_softmax import DISTS
        else:
            from nerf_qa.DISTS_pytorch.DISTS_pt_original import DISTS

        self.dists_model = DISTS()
        self.b1 = nn.Parameter(torch.tensor([params[0]], dtype=torch.float32))
        self.b2 = nn.Parameter(torch.tensor([params[1]], dtype=torch.float32))
        self.b3 = nn.Parameter(torch.tensor([params[2]], dtype=torch.float32))
        self.b4 = nn.Parameter(torch.tensor([params[3]], dtype=torch.float32))
          
    
    def logistic(self, x):
        return (self.b1 - self.b2) / (1 + torch.exp(-(x - self.b3) / torch.abs(self.b4))) + self.b2
    

    def forward(self, dist, ref):
        dists_scores = self.dists_model(dist, ref)
        scores = self.logistic(dists_scores)
        return scores, dists_scores

