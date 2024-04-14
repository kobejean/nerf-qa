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
    def __init__(self, train_df):
        super(NeRFQAModel, self).__init__()


        X = train_df['DISTS'].values

        print("X.shape", X.shape)
        y = train_df['MOS'].values  # Response

        if wandb.config.regression_type == 'logistic':
            def logistic(x, beta1, beta2, beta3, beta4):
                return 2.0*(beta1 - beta2) / (1 + np.exp((x - beta3) / np.abs(beta4))) + beta2
    
            # Initial parameter guesses
            beta1_init = 5.0 #np.max(y)
            beta2_init = 1.0 #np.min(y)
            beta3_init = 0.0 #np.mean(X)
            beta4_init = np.max(X)
            params, params_covariance = curve_fit(logistic, X, y, p0=[beta1_init, beta2_init, beta3_init, beta4_init])
            print(f"Params: {params}")
            self.b1 = nn.Parameter(torch.tensor([params[0]], dtype=torch.float32))
            self.b2 = nn.Parameter(torch.tensor([params[1]], dtype=torch.float32))
            self.b3 = nn.Parameter(torch.tensor([params[2]], dtype=torch.float32))
            self.b4 = nn.Parameter(torch.tensor([params[3]], dtype=torch.float32))
        else:
            if wandb.config.regression_type == 'sqrt': 
                # Create a linear regression model to initialize linear layer
                X = np.sqrt(X)
            model = LinearRegression()
            model.fit(X, y)
            # Print the coefficients
            print(f"Coefficient: {model.coef_[0]}")
            print(f"Intercept: {model.intercept_}")
            self.dists_weight = nn.Parameter(torch.tensor([model.coef_[0]], dtype=torch.float32))
            self.dists_bias = nn.Parameter(torch.tensor([model.intercept_], dtype=torch.float32))


        if wandb.config.dists_weight_norm == 'softmax':
            from nerf_qa.DISTS_pytorch.DISTS_pt_softmax import DISTS
        else:
            from nerf_qa.DISTS_pytorch.DISTS_pt_original import DISTS

        self.dists_model = DISTS()
          
    
    def logistic(self, dists_scores):
        return 2.0 * (self.b1 - self.b2) / (1 + torch.exp((dists_scores - self.b3) / torch.abs(self.b4))) + self.b2
    
    def sqrt(self, dists_scores):
        return torch.sqrt(dists_scores) * self.dists_weight + self.dists_bias 
        
    def linear(self, dists_scores):
        return dists_scores * self.dists_weight + self.dists_bias # linear function
    
    def forward(self, dist, ref):
        dists_scores = self.dists_model(dist, ref)

        if wandb.config.regression_type == 'logistic':
            scores = self.logistic(dists_scores)
        elif wandb.config.regression_type == 'sqrt':
            scores = self.sqrt(dists_scores)
        else:
            scores = self.linear(dists_scores)

        return scores, dists_scores

