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
        self.refine_scale = wandb.config.refine_scale

        # Initialize convolutional layers
        convolutional_layers = [ConvLayer(input_channels, input_channels) for _ in range(depth - 1)]
        convolutional_layers.append(ConvLayer(input_channels, output_channels, apply_relu=False))
        self.block = nn.Sequential(*convolutional_layers)

        # Initialize upsampling layer if upsampling is enabled
        if self.upsample:
            self.upsample_layer = nn.ConvTranspose2d(output_channels, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, input_feats, additional_feats):
        """
        Forward pass of the RefineUp module.

        Parameters:
        - input_feats: The input feature map.
        - additional_feats: Additional feature channels to integrate.

        Returns:
        - Tuple of the refined (and optionally upsampled) feature map and updated additional features.
        """
        # Integrate additional features into input features
        input_feats[:, :self.feature_channels, :, :] += additional_feats[:, :self.feature_channels, :, :]

        # Apply convolutional blocks
        refined_feats = self.block(input_feats)

        # Update additional features with residual scaling
        additional_feats = self.refine_scale * refined_feats[:, :self.feature_channels, :, :] + additional_feats[:, :self.feature_channels, :, :]

        # Upsample if enabled
        if self.upsample:
            refined_feats = self.upsample_layer(refined_feats)

        return refined_feats, additional_feats
    

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
        dinov2_feats = self.dinov2.forward_feats(render_224)
        return dists_feats + [dinov2_feats]

class NRModel(nn.Module):
    """
    No Reference Model for NeRF Quality Assessment.

    This model predicts distortion features of a reference image using the render image.
    It integrates features from a pre-trained dino v2 model and applies a series of
    RefineUp layers to predict a no-reference score.
    """
    def __init__(self, device='cpu', refine_up_depth = 2):
        super(NRModel, self).__init__()
        self.device = device
        self.refine_up_depth = refine_up_depth
        self.encoder = Encoder(device=device)
        
        # Define the channel dimensions based on the encoder models' embeddings and features
        initial_embed_dim = self.encoder.dinov2.embed_dim
        self.sem_channels = [
            initial_embed_dim, initial_embed_dim, initial_embed_dim,
            initial_embed_dim // 2, initial_embed_dim // 4,
            initial_embed_dim // 8, initial_embed_dim // 16,
        ]
        self.dists_channels = [self.encoder.dists.channels[-1]] + list(reversed(self.encoder.dists.channels))

        
        # Initialize decoder
        num_upscales = len(self.dists_channels) - 3
        self.decoder = nn.Sequential(
            *[self.create_refineup_layer(i, upsample=i < num_upscales) for i in range(num_upscales + 2)]
        ).to(self.device)

    def create_refineup_layer(self, index, upsample=True):
        """
        Creates a RefineUp layer for the decoder.
        """
        dists_ch_in, sem_ch_in = self.dists_channels[index], self.sem_channels[index]
        dists_ch_out, sem_ch_out = self.dists_channels[index + 1], self.sem_channels[index + 1]
        channels_in = dists_ch_in + sem_ch_in
        channels_out = dists_ch_out + sem_ch_out
        return RefineUp(channels_in, channels_out, dists_ch_out, depth=self.refine_up_depth, upsample=upsample)


    def pred_gt_dists_feats(self, dists_feats, dinov2_feats):
        # Initialize the feature map by concatenating a zero tensor with the DINO v2 features
        feature_map = torch.concat([torch.zeros_like(dists_feats[-1]), dinov2_feats], dim=1)
        predicted_feats = []

        # Refine features in reverse order using the decoder layers
        for refiner, feature in zip(self.decoder, reversed(dists_feats)):
            feature_map, refined_feature = refiner(feature_map, feature)
            predicted_feats.append(refined_feature)

        # Return the refined features in the original order
        return list(reversed(predicted_feats))
    
    def forward_from_feats(self, encoder_feats):
        # Separate DISTS features and DINO v2 features from the features list
        dists_feats = [feature.to(self.device) for feature in encoder_feats[:-1]]
        dinov2_feats = encoder_feats[-1]['x_norm_patchtokens'].permute(0, 2, 1).reshape(-1, self.encoder.dinov2.embed_dim, 16, 16).to(self.device)
        
        # Predict ground truth DISTS features using the decoded features
        predicted_gt_feats = self.pred_gt_dists_feats(dists_feats, dinov2_feats)
        
        # Compute the final score using DISTS with the predicted features
        return self.encoder.dists.forward_from_feats(dists_feats, predicted_gt_feats)


    def losses(self, gt_image, render, score):
        # Encode render to extract features and predict ground truth DISTS features
        encoder_feats = self.encoder(render)
        dists_feats = [feature.to(self.device) for feature in encoder_feats[:-1]]
        dinov2_feats = encoder_feats[-1]['x_norm_patchtokens'].permute(0, 2, 1).reshape(-1, self.encoder.dinov2.embed_dim, 16, 16).to(self.device)
        predicted_gt_feats = self.pred_gt_dists_feats(dists_feats, dinov2_feats)
        
        # Predict scores and calculate the loss with the ground truth image features
        predicted_score = self.encoder.dists.forward_from_feats(dists_feats, predicted_gt_feats)
        gt_dists_feats = self.encoder.dists.forward_once(gt_image)
        dists_pref2ref = self.encoder.dists.forward_from_feats(predicted_gt_feats, gt_dists_feats, batch_average=True)
        
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
        encoder_feats = self.encoder(render)
        scores = self.forward_from_feats(encoder_feats)
        return scores
