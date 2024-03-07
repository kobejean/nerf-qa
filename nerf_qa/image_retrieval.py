#%%
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class ImageRetrieval:
    def __init__(self, num_words=600):
        self.num_words = num_words
        self.kmeans = None
        self.histograms = None
        self.image_paths = None

    def extract_sift_features(self, images):
        sift = cv2.SIFT_create()
        descriptors_list = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, descriptors = sift.detectAndCompute(gray, None)
            if descriptors is not None:
                descriptors_list.append(descriptors)
        return descriptors_list

    def train_vocabulary(self, descriptors_list):
        all_descriptors = np.vstack(descriptors_list)
        self.kmeans = KMeans(n_clusters=self.num_words, random_state=0).fit(all_descriptors)

    def compute_histograms(self, descriptors_list):
        histograms = []
        for descriptors in descriptors_list:
            histogram = np.zeros(self.num_words)
            if descriptors is not None:
                clusters = self.kmeans.predict(descriptors)
                for i in clusters:
                    histogram[i] += 1
                histogram = histogram / np.linalg.norm(histogram)  # Normalize the histogram
            histograms.append(histogram)
        return np.array(histograms)

    def fit(self, image_paths):
        self.image_paths = image_paths
        images = [cv2.imread(path) for path in image_paths]
        descriptors_list = self.extract_sift_features(images)
        self.train_vocabulary(descriptors_list)
        self.histograms = self.compute_histograms(descriptors_list)

    def retrieve(self, query_image_path, k=5, randomness=0.2):
        query_image = cv2.imread(query_image_path)
        query_descriptors_list = self.extract_sift_features([query_image])
        query_histograms = self.compute_histograms(query_descriptors_list)

        # Compute distances between the query image's histogram and the database histograms
        distances = cdist(query_histograms, self.histograms, metric='cosine')[0]

        # Add randomness to the distances
        random_noise = np.random.uniform(0, randomness, size=len(distances))
        randomized_distances = distances + random_noise

        indices = np.argsort(randomized_distances)[:k]  # Get the indices of k smallest randomized distances

        return [self.image_paths[index] for index in indices]

if __name__ == "__main__":#%%
    # system level
    import os
    from os import path
    import sys
    import argparse


    # deep learning
    from scipy.stats import pearsonr, spearmanr
    import numpy as np
    import torch
    from torch import nn
    from torchvision import models,transforms
    import torch.optim as optim
    import wandb
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LinearRegression
    from torch.utils.data import Dataset, DataLoader, Sampler

    # data 
    import pandas as pd
    import cv2
    from torch.utils.data import TensorDataset
    from tqdm import tqdm
    from PIL import Image
    import plotly.express as px
    import re

    import torch.nn.functional as F
    import numpy as np
    from nerf_qa.roma.utils.utils import tensor_to_pil

    from nerf_qa.roma import roma_indoor
    from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS, prepare_image
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    frame_files_patt = re.compile(r'^\d{3}\.png$')
    def frame_filepaths(dir):
        matches = [f for f in os.listdir(dir) if frame_files_patt.match(f)]
        matches.sort()
        return [path.join(dir, match) for match in matches]

    roma_model = roma_indoor(device=device)
    H, W = roma_model.get_output_resolution()
    dists_model = DISTS().to(device)

    DATA_DIR = "/home/ccl/Datasets/NeRF-QA-Large-1"
    SCORE_FILE = path.join(DATA_DIR, "scores_new.csv")
    # Read the CSV file
    scores_df = pd.read_csv(SCORE_FILE)
    pseudo_dists_scores = []
    pseudo_dists_scores_warp = []
    pseudo_dists_scores_mean = []
    pseudo_dists_scores_warp_mean = []
    for index, row in tqdm(scores_df.iterrows(), total=len(scores_df), position=0):
        ref_filename = row['referenced_filename']
        dist_filename = row['distorted_filename']
        frame_count = row['frame_count']
        
        ref_path = path.join(DATA_DIR, 'references', ref_filename)
        dist_path = path.join(DATA_DIR, 'nerf-renders', dist_filename)
        pseudo_ref_path = path.join(DATA_DIR, 'pseudo-reference', dist_filename)
        image_retrieval_path = path.join(DATA_DIR, 'image_retrieval', dist_filename)
        pseudo_ref_vis_path = path.join(DATA_DIR, 'pseudo-reference-vis', dist_filename)
        os.makedirs(pseudo_ref_path, exist_ok=True)
        os.makedirs(pseudo_ref_vis_path, exist_ok=True)
        os.makedirs(image_retrieval_path, exist_ok=True)
        ref_paths = frame_filepaths(ref_path)
        dist_paths = frame_filepaths(dist_path)
        
        ir = ImageRetrieval(num_words=600)
        dist_count = frame_count//2
        ref_count = frame_count - dist_count
        ir.fit(ref_paths[dist_count:])
        dists_scores = []
        dists_scores_warp = []
        confidence_scores = []

        # Process each frame
        for dist_file in tqdm(dist_paths[:dist_count], position=0, total=frame_count, leave=False):        
            retrieved_images = ir.retrieve(dist_file, k=1)
            print(retrieved_images)

            im1 = Image.open(dist_file).convert('RGB')
            im2 = Image.open(retrieved_images[0]).convert('RGB')
            save_image_retrieval_path = path.join(image_retrieval_path, path.basename(dist_file))
    
            im2.save(save_image_retrieval_path)
            
            true_im = Image.open(path.join(ref_path, path.basename(dist_file))).convert('RGB')
            original_width, original_height = im1.size
            # Match
            with torch.no_grad():
                warp, certainty = roma_model.match(dist_file, retrieved_images[0], device=device)
            save_path_warp = path.join(pseudo_ref_path, path.basename(dist_file)+'.pt')
            
            # Sampling not needed, but can be done with model.sample(warp, certainty)
            x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
            x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)
            true_im = (torch.tensor(np.array(true_im)) / 255).to(device).permute(2, 0, 1)
            warp = warp[:,:W,2:].permute(2,0,1).unsqueeze(0)
            confidence_score = certainty[:,:W].mean()
            certainty = certainty.unsqueeze(0).unsqueeze(0)
            torch.save({
                "warp": warp,
                "certainty": certainty[:,:,:,:W],
            }, save_path_warp)

            distorted_image = prepare_image(tensor_to_pil(x1), resize=True)
            referenced_image = prepare_image(tensor_to_pil(x2), resize=True)
            with torch.no_grad():
                dists_score_warp = dists_model(distorted_image.to(device), referenced_image.to(device), require_grad=False, batch_average=False, warp=warp, certainty=certainty[:,:,:,:W])
                dists_scores_warp.append(dists_score_warp.cpu().item())

            warp = F.interpolate(warp, size=(original_height, original_width), mode='bilinear', align_corners=False).permute(0,2,3,1)
            certainty = F.interpolate(certainty, size=(original_height, original_width*2), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            print(certainty.shape)
            print(warp.shape)

            warp_im = F.grid_sample(
            x2[None], warp, mode="bilinear", align_corners=False
            )[0]
            print(warp_im.shape, x1.shape, x2.shape)
            #certainty = (certainty > 0.1).float()
            warp_im_masked = certainty[:,:original_width] * warp_im + (1 - certainty[:,:original_width])
            x1_masked = certainty[:,:original_width] * x1 + (1 - certainty[:,:original_width])
            x2_masked = certainty[:,original_width:] * x2 + (1 - certainty[:,original_width:])

            distorted_image = prepare_image(tensor_to_pil(x1_masked), resize=True)
            referenced_image = prepare_image(tensor_to_pil(x2_masked), resize=True)
            
            with torch.no_grad():
                dists_score = dists_model(distorted_image.to(device), referenced_image.to(device), require_grad=False, batch_average=False)
            print(dists_score, confidence_score)
            dists_scores.append(dists_score.cpu().item())
            confidence_scores.append(confidence_score.item())

            vis_im_r1 = torch.cat([warp_im_masked,x1],dim=2)
            vis_im_r2 = torch.cat([true_im,x2],dim=2)
            vis_im_r3 = torch.cat([warp_im,x2_masked],dim=2)
            vis_im = torch.cat([vis_im_r1,vis_im_r2,vis_im_r3],dim=1)
            save_path = path.join(pseudo_ref_path, path.basename(dist_file))
            save_path_vis = path.join(pseudo_ref_vis_path, path.basename(dist_file))
            
            tensor_to_pil(warp_im, unnormalize=False).save(save_path)
            tensor_to_pil(vis_im, unnormalize=False).save(save_path_vis)
            #display(tensor_to_pil(vis_im, unnormalize=False).resize((original_height//2, original_width//2)))

        pseudo_dists_score = np.average(dists_scores, weights=confidence_scores)
        pseudo_dists_score_warp = np.average(dists_scores_warp, weights=confidence_scores)
        print(pseudo_dists_score_warp, pseudo_dists_score, row['DISTS'])
        pseudo_dists_scores.append(pseudo_dists_score)
        pseudo_dists_scores_warp.append(pseudo_dists_score_warp)
        pseudo_dists_scores_mean.append(np.mean(dists_scores))
        pseudo_dists_scores_warp_mean.append(np.mean(dists_scores_warp))

        # %%
        scores_df['DISTS_pseudo'] = pseudo_dists_scores
        scores_df['DISTS_pseudo_warp'] = pseudo_dists_scores_warp
        scores_df['DISTS_pseudo_mean'] = pseudo_dists_scores_mean
        scores_df['DISTS_pseudo_warp_mean'] = pseudo_dists_scores_warp_mean
        # %%
        print(np.corrcoef(scores_df['DISTS_pseudo'].values, y=scores_df['DISTS'].values))
        print(np.corrcoef(scores_df['DISTS_pseudo_warp'].values, y=scores_df['DISTS'].values))
        print(np.corrcoef(scores_df['DISTS_pseudo_mean'].values, y=scores_df['DISTS'].values))
        print(np.corrcoef(scores_df['DISTS_pseudo_warp_mean'].values, y=scores_df['DISTS'].values))
        # %%
        from scipy.stats import pearsonr, spearmanr, kendalltau

        def compute_correlations(pred_scores, mos):
            plcc = pearsonr(pred_scores, mos)[0]
            srcc = spearmanr(pred_scores, mos)[0]
            ktcc = kendalltau(pred_scores, mos)[0]

            return {
                'plcc': plcc,
                'srcc': srcc,
                'ktcc': ktcc,
            }
        print(compute_correlations(scores_df['DISTS_pseudo'].values, scores_df['MOS'].values))
        print(compute_correlations(scores_df['DISTS_pseudo_warp'].values, scores_df['MOS'].values))
        print(compute_correlations(scores_df['DISTS_pseudo_mean'].values, scores_df['MOS'].values))
        print(compute_correlations(scores_df['DISTS_pseudo_warp_mean'].values, scores_df['MOS'].values))
        print(compute_correlations(scores_df['DISTS'].values, scores_df['MOS'].values))
        # %%
        scores_df.to_csv(path.join(DATA_DIR, "scores_pseudo.csv"))

        # %%

# %%
