#%%
import cv2
import os
from os import path
import pandas as pd
import cv2
import torch

#%%


import os
import cv2
from os import path

def resize_frame(frame, width, height, keep_aspect_ratio):
    if keep_aspect_ratio:
        # Get current dimensions
        h, w = frame.shape[:2]
        # Calculate the target dimensions
        scale = min(width/w, height/h)
        # Resize while keeping aspect ratio
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Create a new canvas with the target dimensions
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # Compute center offset
        x_center = (width - new_w) // 2
        y_center = (height - new_h) // 2
        # Place the resized frame in the center of the canvas
        canvas[y_center:y_center+new_h, x_center:x_center+new_w] = resized_frame
        return canvas
    else:
        # Resize without keeping aspect ratio
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def convert_frames(video_path, keep_aspect_ratio=False):
    images_path = path.splitext(video_path)[0]
    if keep_aspect_ratio:
        images_path = path.join(images_path, '256_aspect_ratio')
    else:
        images_path = path.join(images_path, '256x256')
    os.makedirs(images_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame before saving
        resized_frame = resize_frame(frame, 256, 256, keep_aspect_ratio)
        nerf_image_path = os.path.join(images_path, f"{i:03d}.png")
        cv2.imwrite(nerf_image_path, resized_frame)
        i += 1
    cap.release()

TEST_DATA_DIR = "/home/ccl/Datasets/NeRF-QA"
TEST_SCORE_FILE = path.join(TEST_DATA_DIR, "NeRF_VQA_MOS.csv")
test_df = pd.read_csv("/Users/kobejean/Developer/GitHub/NeRF-QA-Database/NeRF-QA/NeRF_VQA_MOS.csv")
test_df['scene'] = test_df['reference_filename'].str.replace('_reference.mp4', '', regex=False)
test_size = test_df.shape[0]

ref_dir = path.join(TEST_DATA_DIR, "Reference")
syn_dir = path.join(TEST_DATA_DIR, "NeRF-QA_videos")
for i, row in test_df.iterrows():
    print(syn_dir, row)
    nerf_video_path = path.join(syn_dir, row['distorted_filename'])
    ref_video_path = path.join(ref_dir, row['reference_filename'])
    convert_frames(ref_video_path)
    convert_frames(nerf_video_path)

