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
from nerf_qa.layers import NestedTensorBlock as Block, MemEffAttention
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
class ConvLayer(nn.Module):
    def __init__(self, in_chns, out_chns, apply_relu=True):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_chns, out_chns, kernel_size = 3, stride=1, padding='same', dilation=1, groups=1, bias=True)
        self.batch_norm = nn.BatchNorm2d(out_chns)
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
    - input_chns: Number of channels in the input feature map.
    - output_chns: Number of channels in the output feature map.
    - feature_chns: Number of feature channels to refine.
    - depth: Number of convolutional layers to apply.
    - upsample: Flag indicating whether to upsample the output.
    """
    def __init__(self, input_chns, output_chns, feature_chns, depth=3, upsample=True):
        super(RefineUp, self).__init__()
        self.input_chns = input_chns
        self.output_chns = output_chns
        self.feature_chns = feature_chns
        self.upsample = upsample

        # Initialize convolutional layers
        convolutional_layers = [ConvLayer(input_chns, input_chns) for _ in range(depth - 1)]
        convolutional_layers.append(ConvLayer(input_chns, input_chns, apply_relu=False))
        self.block = nn.Sequential(*convolutional_layers)

        # Initialize upsampling layer if upsampling is enabled
        if self.upsample:
            self.upsample_layer = nn.ConvTranspose2d(input_chns, output_chns, kernel_size=3, stride=2, padding=1, output_padding=1)
        else:
            self.upsample_layer = ConvLayer(input_chns, output_chns, apply_relu=False)
        

    def forward(self, input_feats, additional_feats, trans_decode):
        """
        Forward pass of the RefineUp module.

        Parameters:
        - input_feats: The input feature map.
        - additional_feats: Additional feature channels to integrate.

        Returns:
        - Tuple of the refined (and optionally upsampled) feature map and updated additional features.
        """
        # Integrate additional features into input features
        input_feats = input_feats * wandb.config.refine_scale1
        input_feats[:, :self.feature_chns, :, :] += additional_feats[:, :self.feature_chns, :, :]
        S = input_feats.shape[1] - self.feature_chns
        input_feats[:, -S:, :, :] += F.interpolate(trans_decode[:, -S:, :, :], size=(input_feats.shape[2], input_feats.shape[3]), mode='bilinear', align_corners=True)


        # Apply convolutional blocks
        refined_feats = self.block(input_feats)

        # Update additional features with residual scaling
        additional_feats = wandb.config.refine_scale2 * refined_feats[:, :self.feature_chns, :, :] + additional_feats[:, :self.feature_chns, :, :]

        # Upsample if enabled
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
        self.semantic_model = torch.hub.load("mhamilton723/FeatUp", 'dinov2')

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
    def __init__(self, device='cpu', refine_up_depth = 2):
        super(NRModel, self).__init__()
        self.device = device
        self.refine_up_depth = refine_up_depth
        self.encoder = Encoder(device=device)
        self.l1_loss_fn = nn.L1Loss()
        
        # Define the channel dimensions based on the encoder models' embeddings and features
        initial_sem_dim = self.encoder.dinov2.embed_dim
        initial_dists_dim = self.encoder.dists.chns[-1]
        self.sem_chns = [
            initial_sem_dim, initial_sem_dim,
            initial_sem_dim // 2, initial_sem_dim // 4,
            initial_sem_dim // 8, initial_sem_dim // 16
        ]
        self.dists_chns = list(reversed(self.encoder.dists.chns))
        last_chns = self.sem_chns[-1] + self.dists_chns[-1]
        
        # Initialize decoder
        num_upscales = len(self.dists_chns) - 2
        if wandb.config.transformer_decoder_depth > 0:
            self.transformer_decoder = nn.Sequential(
                *[Block(initial_dists_dim + initial_sem_dim, 8, attn_class=MemEffAttention) for _ in range(wandb.config.transformer_decoder_depth)],
            )
            self.trans2sem = ConvLayer(initial_dists_dim + initial_sem_dim, initial_sem_dim)
        self.score_reg = nn.Sequential(
            ConvLayer(last_chns, last_chns),
            ConvLayer(last_chns, 4, apply_relu=False)
        )
        self.decoder = nn.Sequential(
            *[self.create_refineup_layer(i, upsample=i < num_upscales) for i in range(num_upscales + 2)]
        )
        self.to(device)

    def create_refineup_layer(self, index, upsample=True):
        """
        Creates a RefineUp layer for the decoder.
        """
        dists_ch_in, sem_ch_in = self.dists_chns[index], self.sem_chns[index]
        if index < len(self.dists_chns)-1:
            dists_ch_out, sem_ch_out = self.dists_chns[index + 1], self.sem_chns[index + 1]
        else:
            dists_ch_out, sem_ch_out = dists_ch_in, sem_ch_in
        channels_in = dists_ch_in + sem_ch_in
        channels_out = dists_ch_out + sem_ch_out
        return RefineUp(channels_in, channels_out, dists_ch_in, depth=self.refine_up_depth, upsample=upsample)
    
    def score_regression(self, feature_map):
        score_map = self.score_reg(feature_map)
        mean = score_map.mean([2,3])
        dists = mean[:,0] * 0.1
        mae_map = (score_map[:,1,:,:] * 0.1 + 0.1)
        if wandb.config.reg_activation == 'linear':
            pred_std = (mean[:,2] * 0.05 + 0.05)
            pred_mean = (mean[:,3] * 0.1 + 0.1)
        elif wandb.config.reg_activation == 'relu':
            pred_std = nn.functional.relu_(mean[:,2] * 0.05 + 0.05)
            pred_mean = nn.functional.relu_(mean[:,3] * 0.1 + 0.1)
        elif wandb.config.reg_activation == 'sigmoid':
            pred_std = nn.functional.sigmoid(mean[:,2] * 1.0 - 3.0)
            pred_mean = nn.functional.sigmoid(mean[:,3] * 0.9 - 2.2)
            
        return dists, mae_map, pred_std, pred_mean 

    def pred_gt_dists_feats(self, dists_feats, dinov2_feats):
        # Initialize the feature map by concatenating a zero tensor with the DINO v2 features

        if wandb.config.transformer_decoder_depth > 0:
            encoder_feats = torch.concat([dists_feats[-1], dinov2_feats], dim=1)
            C = encoder_feats.shape[1]
            trans_decode = self.transformer_decoder(encoder_feats.reshape(-1, C, 256).permute(0,2,1)).permute(0,2,1).reshape(-1, C, 16, 16)
            trans_decode = dinov2_feats + wandb.config.refine_scale4 * self.trans2sem(encoder_feats + wandb.config.refine_scale3 * trans_decode)
        else:
            trans_decode = dinov2_feats
        feature_map = torch.concat([torch.zeros_like(dists_feats[-1]), torch.zeros_like(trans_decode)], dim=1)
        predicted_feats = []

        # Refine features in reverse order using the decoder layers
        for refiner, feature in zip(self.decoder, reversed(dists_feats)):
            feature_map, refined_feature = refiner(feature_map, feature, trans_decode)
            predicted_feats.append(refined_feature)

        # Return the refined features in the original order
        return list(reversed(predicted_feats)), feature_map


    def forward_from_feats(self, encoder_feats):
        # Separate DISTS features and DINO v2 features from the features list
        dists_feats = [feature.to(self.device) for feature in encoder_feats[:-1]]
        dinov2_feats = encoder_feats[-1]['x_norm_patchtokens'].permute(0, 2, 1).reshape(-1, self.encoder.dinov2.embed_dim, 16, 16).to(self.device)
        
        # Predict ground truth DISTS features using the decoded features
        predicted_gt_feats, feature_map = self.pred_gt_dists_feats(dists_feats, dinov2_feats)
        dists_res, _, pred_std, pred_mean = self.score_regression(feature_map)
        # Compute the final score using DISTS with the predicted features
        score =  self.encoder.dists.forward_from_feats(dists_feats, predicted_gt_feats)
        score += wandb.config.score_reg_scale * dists_res
        normalized = (score - pred_mean) / (pred_std+1e-7)
        return score, normalized



    def losses(self, gt_image, render, score_std, score_mean):
        # Encode render to extract features and predict ground truth DISTS features
        with torch.no_grad():
            encoder_feats = self.encoder(render)
        dists_feats = [feature.to(self.device) for feature in encoder_feats[:-1]]
        dinov2_feats = encoder_feats[-1]['x_norm_patchtokens'].permute(0, 2, 1).reshape(-1, self.encoder.dinov2.embed_dim, 16, 16).to(self.device)
        predicted_gt_feats, feature_map = self.pred_gt_dists_feats(dists_feats, dinov2_feats)
        
        # Predict scores and calculate the loss with the ground truth image features
        predicted_score = self.encoder.dists.forward_from_feats(dists_feats, predicted_gt_feats)
        with torch.no_grad():
            gt_dists_feats = self.encoder.dists.forward_once(gt_image)
            gt_dists_score = self.encoder.dists.forward_from_feats(gt_dists_feats, dists_feats, batch_average=False)
        dists_pref2ref = self.encoder.dists.forward_from_feats(predicted_gt_feats, gt_dists_feats, batch_average=True)
        gt_mae = torch.abs(gt_image - render['256x256']).mean([1])
        pred_dists_score, pred_mae, pred_std, pred_mean = self.score_regression(feature_map)
        predicted_score += wandb.config.score_reg_scale * pred_dists_score
        
        # Calculate L1 loss and combine it with the DISTS predicted-reference-to-reference loss
        l1_loss = self.l1_loss_fn(predicted_score, gt_dists_score)
        dists_std_l1 = self.l1_loss_fn(pred_std, score_std)
        dists_mean_l1 = self.l1_loss_fn(pred_mean, score_mean)
        mae_reg_l1_loss = self.l1_loss_fn(pred_mae, gt_mae)
        coeff = wandb.config.dists_pref2ref_coeff
        combined_loss = coeff*dists_pref2ref + (1-coeff) * (l1_loss+mae_reg_l1_loss+dists_std_l1+dists_mean_l1)
        
        return {
            "dists_pref2ref": dists_pref2ref,
            "l1": l1_loss,
            "dists_std_l1": dists_std_l1,
            "dists_mean_l1": dists_mean_l1,
            "mae_reg_l1_loss": mae_reg_l1_loss,
            "combined": combined_loss
        }

    def forward(self, render):
        # Encode the render and predict scores from features
        encoder_feats = self.encoder(render)
        scores, normalized = self.forward_from_feats(encoder_feats)
        return scores, normalized
#%%
# %%
#from featup.adaptive_conv_cuda import cuda_impl
# %%
