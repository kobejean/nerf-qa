#%%
import cv2
import os

# Create an output directory if it doesn't exist
output_dir = 'output/reference/lego'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the video
video_path = '/Users/kobejean/Developer/GitHub/NeRF-QA-Database/Reference Videos/Realistic Synthetic 360/lego_reference.mp4'
cap = cv2.VideoCapture(video_path)

frame_number = 0
while True:
    # Read a new frame
    success, frame = cap.read()
    if not success:
        break  # If no frame is found, end the loop

    # Save the frame as a PNG file
    frame_path = os.path.join(output_dir, f"{frame_number}.png")
    cv2.imwrite(frame_path, frame)
    frame_number += 1

cap.release()
print(f"Done! Extracted {frame_number} frames.")

# %%
# Create an output directory if it doesn't exist
output_dir = 'output/synthetic/tensorf/lego'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the video
video_path = '/Users/kobejean/Developer/GitHub/NeRF-QA-Database/Synthesized Videos/Realistic Synthetic 360/Lego/lego_tensorf.mp4'
cap = cv2.VideoCapture(video_path)

frame_number = 0
while True:
    # Read a new frame
    success, frame = cap.read()
    if not success:
        break  # If no frame is found, end the loop

    # Save the frame as a PNG file
    frame_path = os.path.join(output_dir, f"r_{frame_number}.png")
    cv2.imwrite(frame_path, frame)
    frame_number += 1

cap.release()
print(f"Done! Extracted {frame_number} frames.")
# %%
from PIL import Image
import numpy as np
import math
from torchvision import models,transforms
from DISTS_pytorch import DISTS
D = DISTS()


def psnr(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def prepare_image(image, resize=True):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)

def dists(img1, img2):
    img1 = prepare_image(img1)
    img2 = prepare_image(img2)
    return D(img1, img2)

def metrics_from_paths(a_path, b_path):
    a_img = Image.open(a_path)
    b_img = Image.open(b_path)
    return psnr(a_img, b_img), dists(a_img, b_img)
    


# Load the images
gt_img_path = "output/nerf-qa-examples/gt_199.png"
ref_vid_img_path = "output/nerf-qa-examples/ref_vid_199.png"
tensorf_img_path = "output/nerf-qa-examples/tensorf_199.png"


# %%
metrics_from_paths(gt_img_path, ref_vid_img_path)
# %%

metrics_from_paths(gt_img_path, tensorf_img_path)
# %%

metrics_from_paths(ref_vid_img_path, tensorf_img_path)
# %%
