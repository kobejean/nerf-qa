#%%
import os
import csv
import torch
from tqdm import tqdm
from PIL import Image

from nerf_qa.DISTS_pytorch.DISTS_pt import DISTS, prepare_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dists_model = DISTS().to(device)

DATA_DIR = "/home/ccl/Datasets/NeRF-NR-QA/"  # Specify the path to your DATA_DIR

# CSV file path
csv_file = "/home/ccl/Datasets/NeRF-NR-QA/output.csv"

# CSV headers
headers = ["scene", "method", "gt_dir", "render_dir", "frame_count", "frame_height", "frame_width", "basenames", "DISTS"]

# Initialize data rows
data_rows = []

# Iterate over the directory structure
for root, dirs, files in os.walk(DATA_DIR):
    if "color" in dirs:
        # Extract scene and method from the directory path
        path_parts = root.split(os.sep)
        if len(path_parts) >= 7:
            scene = "_".join(path_parts[5:7])
            method = path_parts[-1]
            if method != "gt-color":
                gt_dir = os.path.join("/", *path_parts[:-1], "gt-color")
                render_dir = os.path.join(root, "color")
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
                for gt_file, render_file in tqdm(zip(gt_files, render_files)):
                    gt_im = prepare_image(Image.open(os.path.join(gt_dir, gt_file)).convert('RGB'), resize=True)
                    render_im = prepare_image(Image.open(os.path.join(render_dir, render_file)).convert('RGB'), resize=True)
                    with torch.no_grad():
                        dists_score = dists_model(render_im.to(device), gt_im.to(device), require_grad=False, batch_average=False)
                        dists_scores.append(dists_score.cpu().item())
                        basenames.append(os.path.basename(gt_file))

                gt_dir = os.path.join(*path_parts[5:-1], "gt-color")
                render_dir = os.path.join(*path_parts[5:], "color")

                # Create a data row
                data_row = [scene, method, gt_dir, render_dir, frame_count, frame_height, frame_width, basenames, dists_scores]
                data_rows.append(data_row)

# Write data rows to the CSV file
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(data_rows)

print(f"CSV file '{csv_file}' created successfully.")
# %%
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Print the DataFrame
display(df)
# %%
print(df.head(5))
# %%
import matplotlib.pyplot as plt
import ast

# Assuming your DataFrame is named 'score_df' with a column named 'DISTS'
# Convert the string representation of lists to actual lists
#df['DISTS'] = df['DISTS'].apply(eval)

# Concatenate the lists in the 'DISTS' column
dists_scores = pd.concat([pd.Series(x) for x in df['DISTS']])

# Display the histogram
plt.figure(figsize=(10, 6))
dists_scores.hist(bins=30)
plt.xlabel('DISTS Score')
plt.ylabel('Frequency')
plt.title('Histogram of DISTS Scores')
plt.show()
# %%
