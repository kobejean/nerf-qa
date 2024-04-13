#%%
# system level
import os
from os import path
import sys


# deep learning
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np
import torch
from torch import nn
from torchvision import models,transforms
import torch.optim as optim
import wandb
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression

# data 
import pandas as pd
import cv2
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

import plotly.graph_objects as go
from scipy.optimize import curve_fit
import numpy as np

# Function to plot regression lines for each scene along with all data points
def plot_with_group_regression(pred_scores, mos, scene_video_ids, unique_videos):
    # Define the logistic function with parameters β1 to β4

    def logistic(x, beta1, beta2, beta3, beta4):
        return -2.0*(beta1 - beta2) / (1 + np.exp(-(x) / np.abs(beta4))) + beta2
    
    # Initial parameter guesses
    y = np.array([mos[vid] for vid in unique_videos])
    x = np.array([pred_scores[vid] for vid in unique_videos])

    beta1_init = 5.0
    beta2_init = 1.0
    beta3_init = 0.0
    beta4_init = np.std(x) / 4
    params, params_covariance = curve_fit(logistic, x, y, p0=[beta1_init, beta2_init, beta3_init, beta4_init])
    print(f"Params: {params}")
    fig = go.Figure()

    # Predict using the fitted model for the scene
    x_range = np.linspace(min(x), max(x), 400)
    y_pred = logistic(x_range, *params)


    # Regression line for the scene
    fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', name=f'Regression', line=dict(color='grey')))

    for i, scene_id in enumerate(iter(scene_video_ids.keys())):
        scene_pred_scores = np.array([pred_scores[vid] for vid in scene_video_ids[scene_id]])
        scene_mos = np.array([mos[vid] for vid in scene_video_ids[scene_id]])

        # Use a unique color for each scene, cycling through the colors list
        color = COLORS[i % len(COLORS)]

        fig.add_trace(go.Scatter(y=scene_mos, x=scene_pred_scores, mode='markers', name=f'Score: Scene {scene_id}', marker_color=color))

    fig.update_layout(title='Logistic Regression per Scene between Predicted Score and MOS',
                      yaxis_title='MOS',
                      xaxis_title='Predicted Score')
    return fig


class MetricCollectionLogger():
    def __init__(self, collection_name, log_fn=lambda *args, **kwargs: wandb.log(*args, **kwargs)):
        self.collection_name = collection_name
        self.log_fn = log_fn
        self.metrics = {}
        self.video_ids = []
        self.scene_ids = []
        self.last_correlations = {}
        self.last_scene_min = {}
        self.last_mse = None
        self.last_loss = None

    def add_entries(self, metrics, video_ids, scene_ids):
        video_ids = np.array(video_ids)
        scene_ids = np.array(scene_ids)
        if video_ids.ndim == 0:
            video_ids = np.expand_dims(video_ids, axis=0)
        if scene_ids.ndim == 0:
            scene_ids = np.expand_dims(scene_ids, axis=0)
        self.video_ids.append(video_ids)
        self.scene_ids.append(scene_ids)

        for key, value in metrics.items():
            value = np.array(value) 
            if value.ndim == 0:
                value = np.expand_dims(value, axis=0)
            if key in self.metrics:
                self.metrics[key].append(value)
            else:
                self.metrics[key] = [value]

    def compute_correlations(self, pred_scores, mos):
        plcc = pearsonr(pred_scores, mos)[0]
        srcc = spearmanr(pred_scores, mos)[0]
        ktcc = kendalltau(pred_scores, mos)[0]

        return {
            'plcc': plcc,
            'srcc': srcc,
            'ktcc': ktcc,
        }
    
    def video_metrics_df(self):
        # Concatenate all collected metrics, video_ids, and scene_ids
        metrics = {key: np.concatenate(self.metrics[key]) for key in self.metrics}
        video_ids = np.concatenate(self.video_ids)
        unique_videos = np.unique(video_ids)
        keys = list(metrics.keys())

        # Aggregate metrics by video_id
        video_averages = {key: {} for key in keys}
        for video_id in unique_videos:
            mask = video_ids == video_id
            for key in keys:
                video_averages[key][video_id] = np.mean(metrics[key][mask])

        if 'mse' in video_averages:
            video_averages['rmse'] = {}
            keys.append('rmse')
            for video_id in unique_videos:
                video_averages['rmse'][video_id] = np.sqrt(video_averages['mse'][video_id])

        data = []
        for key in video_averages.keys():
            for video_id, avg in video_averages[key].items():
                # Find if we already have this video_id in our list
                video_entry = next((item for item in data if item['video_id'] == video_id), None)
                if video_entry is None:
                    # If not, create a new dictionary for this video_id
                    video_entry = {'video_id': video_id}
                    data.append(video_entry)
                # Update this dictionary with the new metric
                video_entry[key] = avg

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)

        # If you want 'video_id' to be the index
        df.set_index('video_id', inplace=True)
        return df
          

    def log_summary(self, step):
        logs = {}
        # Concatenate all collected metrics, video_ids, and scene_ids
        metrics = {key: np.concatenate(self.metrics[key]) for key in self.metrics}
        video_ids = np.concatenate(self.video_ids)
        scene_ids = np.concatenate(self.scene_ids)
        unique_videos = np.unique(video_ids)
        unique_scenes = np.unique(scene_ids)
        keys = list(metrics.keys())

        # Aggregate metrics by video_id
        video_averages = {key: {} for key in keys}
        scene_video_ids = {sid: [] for sid in unique_scenes}
        for video_id in unique_videos:
            mask = video_ids == video_id
            for key in keys:
                video_averages[key][video_id] = np.mean(metrics[key][mask])

            scene_id = scene_ids[mask][0]
            scene_video_ids[scene_id].append(video_id)

        if 'mse' in video_averages:
            video_averages['rmse'] = {}
            keys.append('rmse')
            for video_id in unique_videos:
                video_averages['rmse'][video_id] = np.sqrt(video_averages['mse'][video_id])
        
            
        # Aggregate metrics by scene_id
        scene_averages = {key: {} for key in keys}
        for scene_id in unique_scenes:
            mask = scene_ids == scene_id
            video_ids = scene_video_ids[scene_id]
            for key in keys:
                scene_averages[key][scene_id] = np.mean([video_averages[key][vid] for vid in video_ids])
                    
        # Log average metrics
        for key in keys:
            video_average = np.array([video_averages[key][vid] for vid in unique_videos])
            scene_average = np.array([scene_averages[key][sid] for sid in unique_scenes])
            average_over_videos = np.mean(video_average)
            average_over_scenes = np.mean(scene_average)
            logs.update({
                f"{self.collection_name}/{key}": average_over_videos,
                f"{self.collection_name}/average_over_videos/{key}": average_over_videos,
                f"{self.collection_name}/average_over_scenes/{key}": average_over_scenes,
                f"{self.collection_name}/histogram/{key}": wandb.Histogram(video_average),
                f"{self.collection_name}/histogram_over_videos/{key}": wandb.Histogram(video_average),
                f"{self.collection_name}/histogram_over_scenes/{key}": wandb.Histogram(scene_average),
            })
            
            for scene_id in unique_scenes:
                logs.update({
                    f"{self.collection_name}/scene/{scene_id}/{key}": scene_averages[key][scene_id],
                })
            for video_id in unique_videos:
                logs.update({
                    f"{self.collection_name}/video/{video_id}/{key}": video_averages[key][video_id],
                })

        if 'pred_score' in video_averages and 'mos' in video_averages:
            def log_correlations(pred, gt, name='mos', save_last=True):

                logs.update({ f"{self.collection_name}/plot/{name}/scene_regression": plot_with_group_regression(pred, gt, scene_video_ids, unique_videos) })

                real_scene_ids = ['train', 'm60', 'playground', 'truck', 'fortress', 'horns', 'trex', 'room']
                synth_scene_ids = ['ship', 'lego', 'drums', 'ficus', 'hotdog', 'materials', 'mic', 'chair']

                # Aggregate and compute correlations by scene_id
                scene_correlations = {}
                real_scene_pred = []
                real_scene_mos = []
                synth_scene_pred = []
                synth_scene_mos = []
                for scene_id in unique_scenes:
                    scene_pred = np.array([pred[vid] for vid in scene_video_ids[scene_id]])
                    scene_gt = np.array([gt[vid] for vid in scene_video_ids[scene_id]])

                    if len(scene_pred) > 1:  # Ensure there are at least two videos to compute correlations
                        scene_correlations[scene_id] = self.compute_correlations(scene_pred, scene_gt)
                    if scene_id in real_scene_ids:
                        real_scene_pred.append(scene_pred)
                        real_scene_mos.append(scene_gt)
                    elif scene_id in synth_scene_ids:
                        synth_scene_pred.append(scene_pred)
                        synth_scene_mos.append(scene_gt)
                
                if len(real_scene_pred) > 1:
                    real_scene_pred = np.concatenate(real_scene_pred, axis=0)
                    real_scene_mos = np.concatenate(real_scene_mos, axis=0)
                    real_correlations = self.compute_correlations(real_scene_pred, real_scene_mos)
                    logs.update({ f"{self.collection_name}/correlations/real/{name}/{metric}": value for metric, value in real_correlations.items() })

                if len(synth_scene_pred) > 1:
                    synth_scene_pred = np.concatenate(synth_scene_pred, axis=0)
                    synth_scene_mos = np.concatenate(synth_scene_mos, axis=0)
                    synth_correlations = self.compute_correlations(synth_scene_pred, synth_scene_mos)
                    logs.update({ f"{self.collection_name}/correlations/synthetic/{name}/{metric}": value for metric, value in synth_correlations.items() })

                # Log correlations for each scene
                scene_agg = { 'plcc': [], 'srcc': [], 'ktcc': [], }
                real_scene_agg = { 'plcc': [], 'srcc': [], 'ktcc': [], }
                synth_scene_agg = { 'plcc': [], 'srcc': [], 'ktcc': [], }
                for scene_id, corr_values in scene_correlations.items():
                    logs.update({f"{self.collection_name}/correlations/scene/{scene_id}/{name}/{metric}": value for metric, value in corr_values.items()})
                    for metric, value in corr_values.items():
                        scene_agg[metric].append(np.abs(value))
                        if scene_id in real_scene_ids:
                            real_scene_agg[metric].append(np.abs(value))
                        else:
                            synth_scene_agg[metric].append(np.abs(value))
                
                logs.update({f"{self.collection_name}/correlations/scene_min/{name}/{metric}": np.min(value) for metric, value in scene_agg.items()})
                logs.update({f"{self.collection_name}/correlations/scene_mean/{name}/{metric}": np.mean(value) for metric, value in scene_agg.items()})
                logs.update({f"{self.collection_name}/real/correlations/scene_mean/{name}/{metric}": np.mean(value) for metric, value in real_scene_agg.items()})
                logs.update({f"{self.collection_name}/synthetic/correlations/scene_mean/{name}/{metric}": np.mean(value) for metric, value in synth_scene_agg.items()})

                if len(unique_videos) > 1:
                    # Log correlations over all scenes
                    combined_pred = np.array([pred[vid] for vid in unique_videos])
                    combined_gt = np.array([gt[vid] for vid in unique_videos])
                    correlations = self.compute_correlations(combined_pred, combined_gt)
                    
                    logs.update({ f"{self.collection_name}/correlations/{name}/{metric}": value for metric, value in correlations.items() })

                    if save_last:
                        self.last_correlations = correlations
                        # self.last_scene_min = scene_min
                        if 'mse' in video_averages:
                            video_mse = [video_averages['mse'][vid] for vid in unique_videos]
                            self.last_mse = np.mean(video_mse)
                        if 'loss' in video_averages:
                            video_loss = [video_averages['loss'][vid] for vid in unique_videos]
                            self.last_loss = np.mean(video_loss)

            # Prepare video_pred_scores for correlation calculation
            video_pred_scores = video_averages['pred_score']
            video_mos = video_averages['mos']
            log_correlations(video_pred_scores, video_mos, 'mos', True)

            if 'dmos' in video_averages:
                video_dmos = video_averages['dmos']
                log_correlations(video_pred_scores, video_dmos, 'dmos', False)



        self.log_fn(logs, step=step)
        # Reset metrics, video_ids, and scene_ids after logging
        self.metrics = {}
        self.video_ids = []
        self.scene_ids = []


if __name__ == '__main__':
    def test_log(log, step):
        print("Test Log:", log, "Step:", step)

    wandb.init(project='nerf-qa-test')
    logger = MetricCollectionLogger("Train Metrics Dict")

    # Test data
    metrics_data = [
        # (metrics, video_ids, scene_ids)
        ({"pred_score": [4.5, 4.7, 4.6], "mse": [3.9, 3.92, 3.91], "mos": [0.6, 0.67, 0.6]}, [1, 1, 1], [101, 101, 101]),
        ({"pred_score": [3.5, 3.6, 3.7], "mse": [0.85, 0.87, 0.86], "mos": [3.6, 3.6, 3.6]}, [2, 2, 2], [101, 101, 101]),
        ({"pred_score": [4.8, 4.9, 5.0], "mse": [0.93, 0.95, 0.94], "mos": [4.79, 4.9, 4.9]}, [3, 3, 3], [102, 102, 102]),
        ({"pred_score": [3.8, 3.9, 4.0], "mse": [0.88, 0.89, 0.90], "mos": [3.9, 3.9, 3.9]}, [4, 4, 4], [102, 102, 102]),
        ({"pred_score": 5, "mse": 0.88, "mos": 3.9}, 4, 102)
    ]

    # Adding entries to the logger
    for data in metrics_data:
        metrics, video_ids, scene_ids = data
        logger.add_entries(metrics, video_ids=video_ids, scene_ids=scene_ids)

    # Simulating the end of an epoch
    logger.log_summary(step=1)
    
    wandb.finish()


