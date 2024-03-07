#%%
# deep learning
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from sklearn.linear_model import LinearRegression


# local
from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, apply_relu=True):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=1, padding='same', dilation=1, groups=1, bias=True)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.apply_relu = apply_relu
        if self.apply_relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.apply_relu:
            x = self.relu(x)
        return x
class RefineUp(nn.Module):
    """
    Refine and optionally upsample feature maps.

    This module refines feature maps through a series of convolutional layers and
    optionally upsamples the output. It is designed to refine features by integrating
    additional feature channels and applying depth-wise convolution followed by upsampling.

    Parameters:
    - input_channels: Number of channels in the input feature map.
    - output_channels: Number of channels in the output feature map.
    - feature_channels: Number of feature channels to refine.
    - depth: Number of convolutional layers to apply.
    - upsample: Flag indicating whether to upsample the output.
    """
    def __init__(self, input_channels, output_channels, feature_channels, depth=3, upsample=True):
        super(RefineUp, self).__init__()
        self.feature_channels = feature_channels
        self.upsample = upsample
        self.refine_scale = 0.1  # Scale for residual connections

        # Initialize convolutional layers
        convolutional_layers = [ConvLayer(input_channels, input_channels) for _ in range(depth - 1)]
        convolutional_layers.append(ConvLayer(input_channels, output_channels, apply_relu=False))
        self.block = nn.Sequential(*convolutional_layers)

        # Initialize upsampling layer if upsampling is enabled
        if self.upsample:
            self.upsample_layer = nn.ConvTranspose2d(output_channels, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, input_features, additional_features):
        """
        Forward pass of the RefineUp module.

        Parameters:
        - input_features: The input feature map.
        - additional_features: Additional feature channels to integrate.

        Returns:
        - Tuple of the refined (and optionally upsampled) feature map and updated additional features.
        """
        # Integrate additional features into input features
        input_features[:, :self.feature_channels, :, :] += additional_features[:, :self.feature_channels, :, :]

        # Apply convolutional blocks
        refined_features = self.block(input_features)

        # Update additional features with residual scaling
        additional_features = self.refine_scale * refined_features[:, :self.feature_channels, :, :] + additional_features[:, :self.feature_channels, :, :]

        # Upsample if enabled
        if self.upsample:
            refined_features = self.upsample_layer(refined_features)

        return refined_features, additional_features
    

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


class Encoder(nn.Module):
    """
    No Reference Model for NeRF Quality Assessment.

    This model predicts distortion features of a reference image using the render image.
    It integrates features from a pre-trained dino v2 model and applies a series of
    RefineUp layers to predict a no-reference score.
    """
    def __init__(self, device='cpu'):
        super(Encoder, self).__init__()
        self.device = device
        self.initialize_components()

    def initialize_components(self):
        """
        Initializes the DINO v2 and DISTS models and sets their parameters to not require gradients.
        """
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(self.device)
        freeze_parameters(self.dinov2)

        self.dists = DISTS().to(self.device)
        freeze_parameters(self.dists)

    def forward(self, render):
        """
        Encodes render images to feature maps using DINO v2 and DISTS models.
        """
        render_256, render_224 = render["256x256"].to(self.device), render["224x224"].to(self.device)
        dists_feats = self.dists.forward_once(render_256)
        dinov2_feats = self.dinov2.forward_features(render_224)
        return dists_feats + [dinov2_feats]

class NRModel(nn.Module):
    """
    No Reference Model for NeRF Quality Assessment.

    This model predicts distortion features of a reference image using the render image.
    It integrates features from a pre-trained dino v2 model and applies a series of
    RefineUp layers to predict a no-reference score.
    """
    def __init__(self, device='cpu'):
        super(NRModel, self).__init__()
        self.device = device
        self.encoder = Encoder(device=device)
        self.define_channel_dimensions()
        self.initialize_decoder()



    def define_channel_dimensions(self):
        """
        Defines the channel dimensions based on the encoder models' embeddings and features.
        """
        self.sem_channels = [
            self.dinov2.embed_dim, self.dinov2.embed_dim, self.dinov2.embed_dim,
            self.dinov2.embed_dim // 2, self.dinov2.embed_dim // 4,
            self.dinov2.embed_dim // 8, self.dinov2.embed_dim // 16,
        ]
        self.dists_channels = [self.dists.channels[-1]] + list(reversed(self.dists.channels))

    def initialize_decoder(self):
        """
        Initializes the decoder with RefineUp layers based on channel dimensions.
        """
        num_upscales = len(self.dists_channels) - 3
        self.decoder = nn.Sequential(
            *[self.create_refineup_layer(i, upsample=True if i < num_upscales else False) for i in range(num_upscales + 2)]
        ).to(self.device)

    def create_refineup_layer(self, index, upsample=True):
        """
        Creates a RefineUp layer for the decoder.
        """
        dists_ch_in, sem_ch_in = self.dists_channels[index], self.sem_channels[index]
        dists_ch_out, sem_ch_out = self.dists_channels[index + 1], self.sem_channels[index + 1]
        channels_in = dists_ch_in + sem_ch_in
        channels_out = dists_ch_out + sem_ch_out
        return RefineUp(channels_in, channels_out, dists_ch_out, depth=3, upsample=upsample)


    def pred_gt_dists_feats(self, dists_feats, dinov2_feats):
        # Initialize the feature map by concatenating a zero tensor with the DINO v2 features
        feature_map = torch.concat([torch.zeros_like(dists_feats[-1]), dinov2_feats], dim=1)
        predicted_features = []

        # Refine features in reverse order using the decoder layers
        for refiner, feature in zip(self.decoder, reversed(dists_feats)):
            feature_map, refined_feature = refiner(feature_map, feature)
            predicted_features.append(refined_feature)

        # Return the refined features in the original order
        return list(reversed(predicted_features))
    
    def forward_from_feats(self, features_list):
        # Separate DISTS features and DINO v2 features from the features list
        dists_features = [feature.to(self.device) for feature in features_list[:-1]]
        dinov2_features = features_list[-1]['x_norm_patchtokens'].permute(0, 2, 1).reshape(-1, self.dinov2.embed_dim, 16, 16).to(self.device)
        
        # Predict ground truth DISTS features using the decoded features
        predicted_gt_features = self.pred_gt_dists_feats(dists_features, dinov2_features)
        
        # Compute the final score using DISTS with the predicted features
        return self.dists.forward_from_feats(dists_features, predicted_gt_features)


    def losses(self, gt_image, render, score):
        # Encode render to extract features and predict ground truth DISTS features
        features_list = self.encoder(render)
        dists_features = [feature.to(self.device) for feature in features_list[:-1]]
        dinov2_features = features_list[-1]['x_norm_patchtokens'].permute(0, 2, 1).reshape(-1, self.dinov2.embed_dim, 16, 16).to(self.device)
        predicted_gt_features = self.pred_gt_dists_feats(dists_features, dinov2_features)
        
        # Predict scores and calculate the loss with the ground truth image features
        gt_dists_features = self.dists.forward_once(gt_image)
        predicted_score = self.dists.forward_from_feats(dists_features, predicted_gt_features)
        dists_pref2ref = self.dists.forward_from_feats(predicted_gt_features, gt_dists_features, batch_average=True)
        
        # Calculate L1 loss and combine it with the DISTS predicted-reference-to-reference loss
        l1_loss = self.l1_loss_fn(predicted_score, score)
        combined_loss = dists_pref2ref + l1_loss
        
        return {
            "dists_pref2ref": dists_pref2ref,
            "l1": l1_loss,
            "combined": combined_loss
        }

    def forward(self, render):
        # Encode the render and predict scores from features
        features_list = self.encoder(render)
        scores = self.forward_from_feats(features_list)
        return scores

class NRModel_(nn.Module):
    def __init__(self, device='cpu', from_feats=False):
        super(NRModel_, self).__init__()
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
        refiner_depth = 3
        self.l1_loss_fn = nn.L1Loss()
        
        def block(i, upsample=True):
            dists_chn_in, sem_chn_in = self.dists_chns[i], self.sem_chns[i]
            dists_chn_out, sem_chn_out = self.dists_chns[i+1], self.sem_chns[i+1]
            chn_in = dists_chn_in + sem_chn_in
            chn_out = dists_chn_out + sem_chn_out
            return RefineUp(chn_in, chn_out, dists_chn_out, depth=refiner_depth, upsample=upsample)
        
        num_upscales = len(self.dists_chns)-3
        self.decoder = nn.Sequential(
            *[block(i) for i in range(num_upscales)],
            block(num_upscales, upsample=False),
            block(num_upscales+1, upsample=False),
        ).to(device)
        self.device = device

    def encode(self, render):
        render_256, render_224 = render["256x256"].to(self.device), render["224x224"].to(self.device)
        dists_feats = self.dists.forward_once(render_256)
        dinov2_feats = self.dinov2.forward_features(render_224)
        features_list = dists_feats + [dinov2_feats]
        return features_list
    
    def pred_gt_dists_feats(self, dists_feats, dinov2_feats):
        feature_map = torch.concat([torch.zeros_like(dists_feats[-1]), dinov2_feats], dim=1)
        pred_gt_dists_feats = []

        for refiner, dists_feat in zip(self.decoder, reversed(dists_feats)):
            feature_map, pred_feat = refiner(feature_map, dists_feat)
            pred_gt_dists_feats.append(pred_feat)

        return list(reversed(pred_gt_dists_feats))

    def forward_from_feats(self, features_list):
        dists_feats = [feat.to(self.device) for feat in features_list[:-1]]
        dinov2_feats = features_list[-1]['x_norm_patchtokens'].permute(0,2,1).reshape(-1,self.dinov2.embed_dim,16,16).to(self.device)
        pred_gt_dists_feats = self.pred_gt_dists_feats(dists_feats, dinov2_feats)
        return self.dists.forward_from_feats(dists_feats, pred_gt_dists_feats)

    def losses(self, gt_image, render, score):
        score.to(self.device)
        features_list = self.encode(render)
        dists_feats = [feat.to(self.device) for feat in features_list[:-1]]
        dinov2_feats = features_list[-1]['x_norm_patchtokens'].to(self.device).permute(0,2,1).reshape(-1,self.dinov2.embed_dim,16,16)
        pred_gt_dists_feats = self.pred_gt_dists_feats(dists_feats, dinov2_feats)
        gt_dists_feats = self.dists.forward_once(gt_image)
        pred_score = self.dists.forward_from_feats(dists_feats, pred_gt_dists_feats)

        dists_pref2ref = self.dists.forward_from_feats(pred_gt_dists_feats, gt_dists_feats, batch_average=True)
        l1 = self.l1_loss_fn(pred_score, score)
        combined = dists_pref2ref + l1
        return {
            "dists_pref2ref": dists_pref2ref,
            "l1": l1,
            "combined": combined
        }

    def forward(self, render):
        features_list = self.encode(render)
        scores = self.forward_from_feats(features_list)
        return scores
  

