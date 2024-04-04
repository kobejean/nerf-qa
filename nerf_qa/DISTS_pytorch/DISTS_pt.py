# This is a pytoch implementation of DISTS metric.
# Requirements: python >= 3.6, pytorch >= 1.0

import numpy as np
import os,sys
import torch
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F

class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()

class DISTS(torch.nn.Module):
    def __init__(self, load_weights=True, from_feats=False):
        super(DISTS, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
    
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [3,64,128,256,512,512]
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.alpha.data.normal_(0.1,0.01)
        self.beta.data.normal_(0.1,0.01)
        if load_weights:
            weights = torch.load(os.path.join(sys.prefix,'weights.pt'))

            alpha = weights['alpha']
            beta = weights['beta']
            # # Ensure alpha and beta are flattened and concatenated to form a single weight vector
            # weights_concat = torch.cat([alpha, beta], dim=1)
            # print("torch.min(weights_concat)", torch.min(weights_concat), torch.relu(torch.min(weights_concat)))
            # #weights_concat = weights_concat + torch.relu(torch.min(weights_concat))
            # logits_approx = torch.log(torch.clamp(weights_concat, min=0.0) + 1e-10)
            # print(torch.max(torch.abs(torch.softmax(logits_approx, dim=1) - weights_concat)))
            # alpha_logits, beta_logits = torch.split(logits_approx, [alpha.numel(), beta.numel()], dim=1)

            # self.alpha.data = alpha_logits
            # self.beta.data = beta_logits
            self.alpha.data = torch.clamp(alpha, min=1e-10)
            self.beta.data = torch.clamp(beta, min=1e-10)

        
    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x,h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
    
    def warp(self, features, warp, certainty):
        _, _, H, W = features.shape
        print("warp", H, W)
        warp = F.interpolate(warp, size=(H, W), mode='bilinear', align_corners=False).permute(0,2,3,1)
        certainty = F.interpolate(certainty, size=(H, W), mode='bilinear', align_corners=False)
        print(certainty.shape)
        print(warp.shape)

        features = F.grid_sample(
            features, warp, mode="bilinear", align_corners=False
        )[0]
        return features, certainty

    def forward(self, x, y, require_grad=False, batch_average=False, warp=None, certainty=None):
        if require_grad:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)   
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y) 
        dist1 = 0 
        dist2 = 0 
        c1 = 1e-6
        c2 = 1e-6
        
        # w_concat = torch.cat([self.alpha, self.beta], dim=1)
        # w_softmax = torch.softmax(w_concat, dim=1)
        # alpha, beta = torch.split(w_softmax, self.alpha.shape[1], dim=1)
        # alpha = torch.split(alpha, self.chns, dim=1)
        # beta = torch.split(beta, self.chns, dim=1)
        alpha = self.alpha
        beta = self.beta
        # alpha = torch.relu(self.alpha)
        # beta = torch.relu(self.beta)
        w_sum = alpha.sum() + beta.sum()
        alpha = torch.split(alpha/w_sum, self.chns, dim=1)
        beta = torch.split(beta/w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            if warp is None or certainty is None:
                x_mean = feats0[k].mean([2,3], keepdim=True)
                y_mean = feats1[k].mean([2,3], keepdim=True)

                S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
                dist1 = dist1+(alpha[k]*S1).sum(1,keepdim=True)

                x_var = ((feats0[k]-x_mean)**2).mean([2,3], keepdim=True)
                y_var = ((feats1[k]-y_mean)**2).mean([2,3], keepdim=True)
                xy_cov = (feats0[k]*feats1[k]).mean([2,3],keepdim=True) - x_mean*y_mean
            else:
                feats1[k], certainty_k = self.warp(feats1[k], warp, certainty)
                certainty_sum = certainty_k.sum()
                print("C", certainty_k.shape)
                certainty_k = certainty_k
                x_mean = (feats0[k] * certainty_k).sum([2,3], keepdim=True) / certainty_sum
                y_mean = (feats1[k] * certainty_k).sum([2,3], keepdim=True) / certainty_sum

                S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
                dist1 = dist1+(alpha[k]*S1).sum(1,keepdim=True)

                x_var = (certainty_k*(feats0[k]-x_mean)**2).sum([2,3], keepdim=True) / certainty_sum
                y_var = (certainty_k*(feats1[k]-y_mean)**2).sum([2,3], keepdim=True) / certainty_sum
                xy_cov = (certainty_k*feats0[k]*feats1[k]).sum([2,3], keepdim=True) / certainty_sum - x_mean*y_mean
            
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+(beta[k]*S2).sum(1,keepdim=True)

        score = 1 - (dist1+dist2).squeeze(dim=[1,2,3])
        if batch_average:
            return score.mean()
        else:
            return score


    # def forward_from_s1_s2(self, s1, s2, require_grad=False, batch_average=False, warp=None, certainty=None):

    #     dist1 = 0 
    #     dist2 = 0 
    #     c1 = 1e-6
    #     c2 = 1e-6
    #     w_sum = self.alpha.sum() + self.beta.sum()
    #     alpha = torch.split(self.alpha/w_sum, self.chns, dim=1)
    #     beta = torch.split(self.beta/w_sum, self.chns, dim=1)
    #     for k in range(len(self.chns)):
    #         x_mean = feats0[k].mean([2,3], keepdim=True)
    #         y_mean = feats1[k].mean([2,3], keepdim=True)

    #         S1 = feats1[k]
    #         dist1 = dist1+(alpha[k]*S1).sum(1,keepdim=True)

    #         x_var = ((feats0[k]-x_mean)**2).mean([2,3], keepdim=True)
    #         y_var = ((feats1[k]-y_mean)**2).mean([2,3], keepdim=True)
    #         xy_cov = (feats0[k]*feats1[k]).mean([2,3],keepdim=True) - x_mean*y_mean
            
    #         S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
    #         dist2 = dist2+(beta[k]*S2).sum(1,keepdim=True)

    #     score = 1 - (dist1+dist2).squeeze()
    #     if batch_average:
    #         return score.mean()
    #     else:
    #         return score


    def forward_from_feats(self, feats0, feats1, batch_average=False):

        dist1 = 0 
        dist2 = 0 
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha/w_sum, self.chns, dim=1)
        beta = torch.split(self.beta/w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2,3], keepdim=True)
            y_mean = feats1[k].mean([2,3], keepdim=True)

            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            dist1 = dist1+(alpha[k]*S1).sum(1,keepdim=True)

            x_var = ((feats0[k]-x_mean)**2).mean([2,3], keepdim=True)
            y_var = ((feats1[k]-y_mean)**2).mean([2,3], keepdim=True)
            xy_cov = (feats0[k]*feats1[k]).mean([2,3],keepdim=True) - x_mean*y_mean
   
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+(beta[k]*S2).sum(1,keepdim=True)

        score = 1 - (dist1+dist2).squeeze([1,2,3])
        if batch_average:
            return score.mean()
        else:
            return score

def prepare_image(image, resize=True, keep_aspect_ratio=False):
    if resize and min(image.size)>256:
        if keep_aspect_ratio:
            image = transforms.functional.resize(image,256)
        else:
            image = transforms.functional.resize(image,(256, 256))
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)


if __name__ == '__main__':

    from PIL import Image
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='../images/r0.png')
    parser.add_argument('--dist', type=str, default='../images/r1.png')
    args = parser.parse_args()
    
    ref = prepare_image(Image.open(args.ref).convert("RGB"))
    dist = prepare_image(Image.open(args.dist).convert("RGB"))
    assert ref.shape == dist.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DISTS().to(device)
    ref = ref.to(device)
    dist = dist.to(device)
    score = model(ref, dist)
    print(score.item())
    # score: 0.3347

