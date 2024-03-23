#%%
import os
import csv
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS, prepare_image
from nerf_qa.ADISTS import ADISTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
dists_model = ADISTS().to(device)

DATA_DIR = "/home/ccl/Datasets/NeRF-NR-QA/"  # Specify the path to your DATA_DIR

# CSV file path
csv_file = "/home/ccl/Datasets/NeRF-NR-QA/output_ADISTS.csv"

# CSV headers
headers = ["scene", "method", "gt_dir", "render_dir", "frame_count", "frame_height", "frame_width", "basenames", "score_map_log_min", "score_map_log_max"]

# Initialize data rows
data_rows = []


def load_image(path):
    image = Image.open(path)

    if image.mode == 'RGBA':
        # If the image has an alpha channel, create a white background
        background = Image.new('RGBA', image.size, (255, 255, 255))
        
        # Paste the image onto the white background using alpha compositing
        background.paste(image, mask=image.split()[3])
        
        # Convert the image to RGB mode
        image = background.convert('RGB')
    else:
        # If the image doesn't have an alpha channel, directly convert it to RGB
        image = image.convert('RGB')
    return image

# Iterate over the directory structure
for root, dirs, files in os.walk(DATA_DIR):
    if "color" in dirs or "gt-color" in dirs:
        # Extract scene and method from the directory path
        path_parts = root.split(os.sep)
        if len(path_parts) >= 7:
            scene = "_".join(path_parts[5:7])
            method = path_parts[-1]
            #if method != "gt-color":
            if True:
                if "color" in dirs:
                    gt_dir = os.path.join("/", *path_parts[:-1], "gt-color")
                    render_dir = os.path.join(root, "color")
                    score_map_dir = os.path.join(root, "score-map")
                else:
                    gt_dir = render_dir = os.path.join(root, 'gt-color')
                    score_map_dir = os.path.join(root, "gt-score-map")
                    method = 'gt'
                os.makedirs(score_map_dir, exist_ok=True)
                print(gt_dir, render_dir)

                # Count the number of image files in gt_dir or render_dir
                gt_files = [f for f in os.listdir(gt_dir) if f.endswith((".jpg", ".png"))]
                gt_files.sort()
                render_files = [f for f in os.listdir(render_dir) if f.endswith((".jpg", ".png"))]
                render_files.sort()
                frame_count = max(len(gt_files), len(render_files))

                # Get frame dimensions from an image file
                if gt_files:
                    image_path = os.path.join(gt_dir, gt_files[0])
                elif render_files:
                    image_path = os.path.join(render_dir, render_files[0])
                else:
                    continue

                with Image.open(image_path) as img:
                    frame_width, frame_height = img.size
                dists_scores = []
                basenames = []
                score_maps = []
                for gt_file, render_file in tqdm(zip(gt_files, render_files)):
                    gt_im = prepare_image(load_image(os.path.join(gt_dir, gt_file)), resize=False)
                    render_im = prepare_image(load_image(os.path.join(render_dir, render_file)), resize=False)
                    
                    h, w = (int(render_im.shape[2]*0.7), int(render_im.shape[3]*0.7))
                    i, j = (render_im.shape[2]-h)//2, (render_im.shape[3]-w)//2
                    # Crop to avoid black region due to postprocessed distortion
                    render_im = TF.crop(render_im, i, j, h, w)
                    gt_im = TF.crop(gt_im, i, j, h, w)
                    #render_im = TF.resize(render_im,(256, 256))
                    #gt_im = TF.resize(gt_im,(256, 256))

                    with torch.no_grad():
                        dists_score = dists_model(render_im.to(device), gt_im.to(device), as_map=True)
                        #dists_scores.append(dists_score.cpu().item())
                        basename = os.path.basename(gt_file)
                        basenames.append(basename)
                        score_map_path = os.path.splitext(basename)[0] + '.pt'
                        score_map_path = os.path.join(score_map_dir, os.path.splitext(basename)[0] + '.pt')
                        if os.path.exists(score_map_path):
                            os.remove(score_map_path)

                        # Ensure tensor is on CPU and is a float32 tensor
                        dists_score = dists_score.detach().cpu()
                        dists_scores.append(dists_score.mean().item())
                        score_maps.append(torch.clamp(dists_score, min=1e-10, max=1.0))
                score_maps = torch.concat(score_maps, dim=0)
                score_maps_min = score_maps.amin(dim=[2,3]).squeeze(1)
                score_maps_max = score_maps.amax(dim=[2,3]).squeeze(1)
                score_log_max = (-torch.log10(score_maps_min)).numpy().tolist()
                score_log_min = (-torch.log10(score_maps_max)).numpy().tolist()
                
                for i, basename in tqdm(enumerate(basenames)):
                    score_map_path = os.path.join(score_map_dir, basename)
                    log_min = score_log_min[i]
                    log_max = score_log_max[i]
                    dists_score = -torch.log10(score_maps[i])
                    spread = (log_max-log_min)
                    dists_score = 255 * (dists_score-log_min)/spread if spread > 0 else torch.zeros_like(dists_score)
                    dists_score = dists_score.byte() # quantize
                    dists_score = dists_score.squeeze([0])
                    image = Image.fromarray(dists_score.numpy(), mode='L')
                    image.save(score_map_path, format='PNG')

                if "color" in dirs:
                    gt_dir = os.path.join(*path_parts[5:-1], "gt-color")
                    render_dir = os.path.join(*path_parts[5:], "color")
                else:
                    gt_dir = render_dir = os.path.join(*path_parts[5:], 'gt-color')
                print(gt_dir, render_dir)
                # Create a data row
                data_row = [scene, method, gt_dir, render_dir, frame_count, frame_height, frame_width, basenames, dists_scores, score_log_min, score_log_max]
                data_rows.append(data_row)

# Write data rows to the CSV file
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(data_rows)

print(f"CSV file '{csv_file}' created successfully.")
# %%
import pandas as pd

DATA_DIR = "/home/ccl/Datasets/NeRF-NR-QA/"  # Specify the path to your DATA_DIR

# CSV file path
csv_file = "/home/ccl/Datasets/NeRF-NR-QA/output.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# %%
import matplotlib.pyplot as plt
import ast
import numpy as np

# Assuming your DataFrame is named 'score_df' with a column named 'DISTS'
# Convert the string representation of lists to actual lists
df['DISTS_list'] = df['DISTS'].apply(eval)

# Concatenate the lists in the 'DISTS' column
dists_scores = pd.concat([pd.Series(x) for x in df['DISTS_list']])

# Display the histogram
plt.figure(figsize=(10, 6))
dists_scores.hist(bins = np.arange(-0.01, 0.91, 0.01))
plt.xlabel('DISTS Score')
plt.ylabel('Frequency')
plt.title('Histogram of DISTS Scores')
plt.show()
# %%

# %%
