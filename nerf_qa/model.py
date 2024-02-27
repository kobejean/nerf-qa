# deep learning
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import wandb
from sklearn.linear_model import LinearRegression


# local
from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NeRFQAModel(nn.Module):
    def __init__(self, train_df, linearization_type = True):
        super(NeRFQAModel, self).__init__()
        self.linearization_type = linearization_type
        # Reshape data (scikit-learn expects X to be a 2D array)
        X = train_df['DISTS'].values.reshape(-1, 1)  # Predictor
        y = train_df['MOS'].values  # Response

        # Create a linear regression model to initialize linear layer
        model = LinearRegression()
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
            scores = self.dists_model(ref_images, dist_images, require_grad=False, batch_average=False)  # Returns a tensor of scores
            
            # Collect scores tensors
            all_scores.append(scores)

        # Concatenate all score tensors into a single tensor
        all_scores_tensor = torch.cat(all_scores, dim=0)

        # Compute the average score across all batches
        average_score = torch.mean(all_scores_tensor) if all_scores_tensor.numel() > 0 else torch.tensor(0.0).to(device)

        return average_score
        
    def forward_dataloader(self, dataloader):
        raw_scores = self.compute_dists_with_batches(dataloader)
        
        # Normalize raw scores using the trainable mean and std
        normalized_scores = raw_scores * self.dists_weight.detach() + self.dists_bias.detach()
        return normalized_scores

    def forward(self, dist, ref):
        scores = self.dists_model(ref, dist, require_grad=True, batch_average=False)  # Returns a tensor of scores
        scores = scores * self.dists_weight + self.dists_bias # linear function
        return scores
    
