#%%
# deep learning
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from sklearn.linear_model import LinearRegression

from featup.layers import ChannelNorm
# local
from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS
from nerf_qa.layers import NestedTensorBlock as Block, MemEffAttention
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
class ConvLayer(nn.Module):
    def __init__(self, in_chns, out_chns, activation_enabled=True):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(p=0.2)
        self.conv = nn.Conv2d(in_chns, out_chns, kernel_size = 3, stride=1, padding='same', dilation=1, groups=1, bias=True)
        self.norm_layer = ChannelNorm(out_chns)
        self.activation_enabled = activation_enabled
        if self.activation_enabled:
            self.act_layer = nn.GELU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.norm_layer(x)
        if self.activation_enabled:
            x = self.act_layer(x)
        return x

class ConvTransposeLayer(nn.Module):
    def __init__(self, in_chns, out_chns, activation_enabled=True):
        super(ConvTransposeLayer, self).__init__()
        self.dropout = nn.Dropout2d(p=0.2)
        self.conv = nn.ConvTranspose2d(in_chns, out_chns, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm_layer = ChannelNorm(out_chns)
        self.activation_enabled = activation_enabled
        if self.activation_enabled:
            self.act_layer = nn.GELU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.norm_layer(x)
        if self.activation_enabled:
            x = self.act_layer(x)
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
        convolutional_layers.append(ConvLayer(input_chns, input_chns, activation_enabled=False))
        self.block = nn.Sequential(*convolutional_layers)

        # Initialize upsampling layer if upsampling is enabled
        if self.upsample:
            self.upsample_layer = ConvTransposeLayer(input_chns, output_chns, activation_enabled=False)
        else:
            self.upsample_layer = ConvLayer(input_chns, output_chns, activation_enabled=False)
        

    def forward(self, input_feats, dists_feat, sem_feat):
        # Integrate additional features into input features
        input_feats = input_feats * wandb.config.refine_scale1 + torch.concat([dists_feat, sem_feat], dim=1)

        # Apply convolutional blocks
        feature_map = self.block(input_feats)

        # Update additional features with residual scaling
        pred_feats = wandb.config.refine_scale2 * feature_map[:, :self.feature_chns, :, :] + dists_feat

        # Upsample if enabled
        feature_map = self.upsample_layer(feature_map)
        return feature_map, pred_feats
    

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

    
class SemanticEncoder(nn.Module):
    def __init__(self, model_name='dinov2'):
        super(SemanticEncoder, self).__init__()
        self.device = device
        featup = torch.hub.load("mhamilton723/FeatUp", model_name)
        self.model = featup.model
        self.upsampler = featup.upsampler
        self.dim = featup.dim

    def upsample(self, feats, image):
        feats_2 = self.upsampler.upsample(feats, image, self.upsampler.up1)
        feats_4 = self.upsampler.upsample(feats_2, image, self.upsampler.up2)
        feats_8 = self.upsampler.upsample(feats_4, image, self.upsampler.up3)
        feats_16 = self.upsampler.upsample(feats_8, image, self.upsampler.up4)

        feats = self.upsampler.fixup_proj(feats) * 0.1 + feats
        feats_2 = self.upsampler.fixup_proj(feats_2) * 0.1 + feats_2
        feats_4 = self.upsampler.fixup_proj(feats_4) * 0.1 + feats_4
        feats_8 = self.upsampler.fixup_proj(feats_8) * 0.1 + feats_8
        feats_16 = self.upsampler.fixup_proj(feats_16) * 0.1 + feats_16
        return [feats, feats_2, feats_4, feats_8, feats_16, feats_16]

    def forward(self, image):
        feats = self.model(image)
        return feats, self.upsample(feats, image)
    
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
        self.semantic_model = SemanticEncoder(wandb.config.vit_model)
        freeze_parameters(self.semantic_model)
        
        self.dists = DISTS()
        freeze_parameters(self.dists)
        self.to(self.device)

    def forward(self, render):
        """
        Encodes render images to feature maps using DINO v2 and DISTS models.
        """
        render_256, render_224 = render["256x256"].to(self.device), render["224x224"].to(self.device)
        sem_feats, sem_feats_upscaled = self.semantic_model(render_224)
        dists_feats = self.dists.forward_once(render_256)
        # multi_scale_feats = list(zip(reversed(dists_feats), sem_feats_upscaled))
        return dists_feats, sem_feats, sem_feats_upscaled

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
        initial_sem_dim = self.encoder.semantic_model.dim
        initial_dists_dim = self.encoder.dists.chns[-1]
        self.sem_chns = [initial_sem_dim] * 6
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
            ConvLayer(last_chns, 4, activation_enabled=False)
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

    def pred_gt_dists_feats(self, encoder_feats):
        dists_feats, sem_feats, sem_feats_upscaled = encoder_feats

        if wandb.config.transformer_decoder_depth > 0:
            encoder_feats = torch.concat([dists_feats[-1], sem_feats], dim=1)
            C = encoder_feats.shape[1]
            trans_decode = self.transformer_decoder(encoder_feats.reshape(-1, C, 256).permute(0,2,1)).permute(0,2,1).reshape(-1, C, 16, 16)
            trans_decode = sem_feats + wandb.config.refine_scale4 * self.trans2sem(encoder_feats + wandb.config.refine_scale3 * trans_decode)
        else:
            trans_decode = sem_feats
        feature_map = torch.concat([dists_feats[-1], trans_decode], dim=1)
        predicted_feats = []

        # Refine features in reverse order using the decoder layers
        for refiner, dists_feat, sem_feat in zip(self.decoder, reversed(dists_feats), sem_feats_upscaled):
            feature_map, refined_feature = refiner(feature_map, dists_feat, sem_feat)
            predicted_feats.append(refined_feature)

        # Return the refined features in the original order
        return list(reversed(predicted_feats)), feature_map


    def forward_from_feats(self, encoder_feats):
        dists_feats, sem_feats, sem_feats_upscaled = encoder_feats
        
        # Predict ground truth DISTS features using the decoded features
        predicted_gt_feats, feature_map = self.pred_gt_dists_feats(encoder_feats)
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
        dists_feats, sem_feats, sem_feats_upscaled = encoder_feats
        predicted_gt_feats, feature_map = self.pred_gt_dists_feats(encoder_feats)
        
        # Predict scores and calculate the loss with the ground truth image features
        predicted_score = self.encoder.dists.forward_from_feats(dists_feats, predicted_gt_feats, batch_average=False)
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
        combined_loss = coeff*dists_pref2ref + (1-coeff) * (l1_loss+0.2*(mae_reg_l1_loss+dists_std_l1+dists_mean_l1))
        
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

