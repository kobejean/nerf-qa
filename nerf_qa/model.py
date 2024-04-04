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


# local
from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def linear_func(x, a, b):
    return a * x + b

class ConvLayer(nn.Module):
    def __init__(self, in_chns, out_chns, activation_enabled=True):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(p=0.2)
        self.conv = nn.Conv2d(in_chns, out_chns, kernel_size = 1, stride=1, padding='same', dilation=1, groups=1, bias=True)
        self.activation_enabled = activation_enabled
        if self.activation_enabled:
            self.act_layer = nn.GELU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        if self.activation_enabled:
            x = self.act_layer(x)
        return x

class NeRFQAModel(nn.Module):
    def __init__(self, train_df):
        super(NeRFQAModel, self).__init__()
        # Reshape data (scikit-learn expects X to be a 2D array)
        # X = train_df['DISTS_no_resize'].values.reshape(-1, 1)  # Predictor

        unique_groups = train_df['scene_type'].unique()
        self.scene_type_to_idx = {group: i for i, group in enumerate(unique_groups)}
        self.num_scene_types = len(unique_groups)

        dists_scene_type_weight = torch.zeros(self.num_scene_types)
        dists_scene_type_bias = torch.zeros(self.num_scene_types)

        for group in unique_groups:
            group_df = train_df[train_df['scene_type'] == group]
            group_x = group_df['DISTS']
            group_y = group_df['MOS']
            params, _ = curve_fit(linear_func, group_x, group_y)
            idx = self.scene_type_to_idx[group]
            dists_scene_type_weight[idx] = params[0]
            dists_scene_type_bias[idx] = params[1]

        dists_scene_type_weight_mean = dists_scene_type_weight.mean()
        dists_scene_type_weight_std = dists_scene_type_weight.std()
        dists_scene_type_bias_mean = dists_scene_type_bias.mean()
        dists_scene_type_bias_std = dists_scene_type_bias.std()
        self.dists_scene_type_weight = nn.Parameter(dists_scene_type_weight)
        self.dists_scene_type_bias = nn.Parameter(dists_scene_type_bias)

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

        self.scene_type_bias_weight = nn.Parameter(torch.tensor([wandb.config.init_scene_type_bias_weight], dtype=torch.float32))
        self.conv = ConvLayer(64, 2, activation_enabled=False)
        self.conv.conv.weight.data[0].normal_(0, torch.sqrt(2. / 64.) * dists_scene_type_weight_std)
        self.conv.conv.weight.data[1].normal_(0, torch.sqrt(2. / 64.) * dists_scene_type_bias_std)
        self.conv.conv.bias.data[0] = dists_scene_type_weight_mean
        self.conv.conv.bias.data[1] = dists_scene_type_bias_mean

    def get_param_lr(self):
        linear_layer_params = [
            self.dists_scene_type_weight,
            self.dists_scene_type_bias,
            self.dists_weight,
            self.dists_bias,
            self.scene_type_bias_weight
        ]
        cnn_layer_params = [
            self.conv.conv.weight,
            self.conv.conv.bias,
        ]

        # Collect the remaining parameters
        # remaining_params = [param for param in self.parameters() if param not in linear_layer_params]
        remaining_params = [param for param in self.parameters() if all(param is not p for p in (linear_layer_params + cnn_layer_params))]
        return  [
            {'params': linear_layer_params, 'lr': wandb.config.linear_layer_lr },  # Set the learning rate for the 
            {'params': cnn_layer_params, 'lr': wandb.config.cnn_layer_lr },  # Set the learning rate for the specific layer
            {'params': remaining_params }  # Set the learning rate for the remaining parameters
        ]

            
    
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
        normalized_scores = raw_scores * self.dists_weight + self.dists_bias
        return normalized_scores

    def forward(self, dist, ref, scene_type = None):
        with torch.no_grad():
            feats0 = self.forward_once(dist)
            feats1 = self.forward_once(ref) 
        # dists_scores = self.dists_model(ref, dist, require_grad=False, batch_average=False)  # Returns a tensor of scores
        complexity = self.conv(feats1[1]).mean([2,3])
        complexity_weight = complexity[:,0]
        complexity_bias = complexity[:,1]
        dists_scores = self.dists_model.forward_from_feats(feats0, feats1)
        
        scores = dists_scores * complexity_weight + complexity_bias # linear function
        # scores = dists_scores * self.dists_weight + self.dists_bias # linear function
        return scores



class NeRFNRQAModel(nn.Module):
    def __init__(self, device='cpu', from_feats=False):
        super(NeRFNRQAModel, self).__init__()
        if not from_feats:
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(device)
            for param in self.dinov2.parameters():
                param.requires_grad = False

        self.dists = DISTS(from_feats=from_feats).to(device)
        for param in self.dists.parameters():
            param.requires_grad = False

        self.sem_chns = [
            self.dinov2.embed_dim,
            self.dinov2.embed_dim,
            self.dinov2.embed_dim,
            self.dinov2.embed_dim//2,
            self.dinov2.embed_dim//4,
            self.dinov2.embed_dim//8,
            self.dinov2.embed_dim//16,
        ]
        self.dists_chns = [self.dists.chns[-1]] + list(reversed(self.dists.chns))
        last_chn_in = self.dists_chns[-2]+self.sem_chns[-2]
        last_chn_out = self.dists_chns[-1]+self.sem_chns[-1]
        
        def block(i):
            dists_chn_in, sem_chn_in = self.dists_chns[i], self.sem_chns[i]
            dists_chn_out, sem_chn_out = self.dists_chns[i+1], self.sem_chns[i+1]
            chn_in = dists_chn_in + sem_chn_in
            chn_out = dists_chn_out + sem_chn_out
            return nn.Sequential(
                    nn.Sequential(nn.Conv2d(chn_in, chn_out, padding='same', kernel_size=3), nn.BatchNorm2d(chn_out), nn.ReLU()),
                    nn.Sequential(nn.Conv2d(chn_out, chn_out, padding='same', kernel_size=3), nn.BatchNorm2d(chn_out), nn.ReLU()),
                    nn.Sequential(nn.Conv2d(chn_out, chn_out, padding='same', kernel_size=3), nn.BatchNorm2d(chn_out)),
                )
        
        def upscale(i):
            dists_chn_out, sem_chn_out = self.dists_chns[i+1], self.sem_chns[i+1]
            chn_out = dists_chn_out + sem_chn_out
            return nn.ConvTranspose2d(chn_out, chn_out, kernel_size=3, stride=2, padding=1, output_padding=1)
        num_upscales = len(self.dists_chns)-3
        self.decoder = nn.Sequential(
            *[nn.Sequential(block(i), upscale(i)) for i in range(num_upscales)],
            block(num_upscales),
            nn.Sequential(
                nn.Sequential(nn.Conv2d(last_chn_in, last_chn_in, padding='same', kernel_size=3), nn.BatchNorm2d(last_chn_in), nn.ReLU()),
                nn.Sequential(nn.Conv2d(last_chn_in, last_chn_in, padding='same', kernel_size=3), nn.BatchNorm2d(last_chn_in), nn.ReLU()),
                nn.Sequential(nn.Conv2d(last_chn_in, last_chn_out, padding='same', kernel_size=3), nn.BatchNorm2d(last_chn_out)),
            ),
        ).to(device)
        self.device = device

    def encode(self, render_256, render_224):
        dists_feats = self.dists.forward_once(render_256)
        dinov2_feats = self.dinov2.forward_features(render_224)
        features_list = dists_feats + [dinov2_feats]
        return features_list

    def forward_from_feats(self, features_list):
        res_scale = 0.1
        dists_feats = [feat.to(self.device) for feat in reversed(features_list[:-1])]
        B = dists_feats[0].shape[0]

        dinov2_feats = features_list[-1]['x_norm_patchtokens'].permute(0,2,1).reshape(B,self.dinov2.embed_dim,16,16).to(self.device)
        
        feature_map = torch.concat([dists_feats[0], dinov2_feats], dim=1)
        pred_feats = []

        for i in range(len(self.decoder)-2):
            block = self.decoder[i][0]
            upsample = self.decoder[i][1]
            dists_chn = self.dists_chns[i+1]

            feature_map[:,:dists_chn,:,:] += dists_feats[i]
            feature_map = block(feature_map)
            pred_feat = feature_map[:,:dists_chn,:,:] * res_scale + dists_feats[i]
            pred_feats.append(pred_feat)

            feature_map = upsample(feature_map)

        block = self.decoder[-2]
        dists_chn = self.dists_chns[-2]
        feature_map[:,:dists_chn,:,:] += dists_feats[-2]
        feature_map = block(feature_map)

        pred_feat = feature_map[:,:dists_chn,:,:] * res_scale + dists_feats[-2]
        pred_feats.append(pred_feat)

        block = self.decoder[-1]
        dists_chn = self.dists_chns[-1]
        feature_map[:,:dists_chn,:,:] += dists_feats[-1]
        feature_map = block(feature_map)

        pred_feat = feature_map[:,:dists_chn,:,:] * res_scale + dists_feats[-1]
        pred_feats.append(pred_feat)

        dists_feats = list(reversed(dists_feats))
        pred_feats = list(reversed(pred_feats))
        return self.dists.forward_from_feats(dists_feats, pred_feats)

    def forward(self, render_256, render_224):
        features_list = self.encode(render_256.to(self.device), render_224.to(self.device))
        scores = self.forward_from_feats(features_list)
        return scores
    
if __name__ == "__main__":
    import os
    import pandas as pd
    from nerf_qa.data import NerfNRQADataset
    from torch.utils.data import Dataset, DataLoader, Sampler
    from tqdm import tqdm
    DATA_DIR = "/home/ccl/Datasets/NeRF-NR-QA/"  # Specify the path to your DATA_DIR
    #%%
    # CSV file 
    scores_df = pd.read_csv("/home/ccl/Datasets/NeRF-NR-QA/output.csv")
    
    dataset = NerfNRQADataset(scores_df, dir = DATA_DIR, mode='gt,render')
    dataloader = DataLoader(dataset, batch_size=1)
    model = NeRFNRQAModel(device=device)
    for gt_im, render_im, score, render_id, frame_id in tqdm(dataloader):
        render_dir = scores_df['render_dir'].iloc[render_id.numpy()].values[0]
        basename = (eval(scores_df['basenames'].iloc[render_id.numpy()].values[0]))[frame_id.numpy()[0]]
        features_list = model.encode(render_im.to(device))# Extract the filename without the extension
        
        filename, _ = os.path.splitext(basename)

        # Construct the save path
        parent_dir = os.path.dirname(render_dir)  # Get the parent directory of 'color'
        features_dir = os.path.join(DATA_DIR, parent_dir, 'features')  # Create the 'features' directory path
        save_path = os.path.join(features_dir, f"{filename}.pt")

        # Create the 'features' directory if it doesn't exist
        os.makedirs(features_dir, exist_ok=True)

        # Save the features_list to the save_path
        #print(save_path, render_id, frame_id)
        torch.save(features_list, save_path)

# %%
