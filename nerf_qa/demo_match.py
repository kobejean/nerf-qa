from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from nerf_qa.roma.utils.utils import tensor_to_pil

from nerf_qa.roma import roma_indoor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="/home/ccl/Datasets/NeRF-QA-Large-1/references/gt_horns/000.png", type=str)
    parser.add_argument("--im_B_path", default="/home/ccl/Datasets/NeRF-QA-Large-1/references/gt_horns/005.png", type=str)
    parser.add_argument("--save_path", default="demo/garden.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    save_path = args.save_path

    # Create model
    roma_model = roma_indoor(device=device)

    H, W = roma_model.get_output_resolution()


    ref_paths = [
        "/home/ccl/Datasets/NeRF-QA-Large-1/references/gt_horns/001.png",
        "/home/ccl/Datasets/NeRF-QA-Large-1/references/gt_horns/002.png",
        "/home/ccl/Datasets/NeRF-QA-Large-1/references/gt_horns/003.png",
        "/home/ccl/Datasets/NeRF-QA-Large-1/references/gt_horns/004.png",
        "/home/ccl/Datasets/NeRF-QA-Large-1/references/gt_horns/005.png",
        "/home/ccl/Datasets/NeRF-QA-Large-1/references/gt_horns/006.png",
    ]

    white_im = torch.ones((H,2*W),device=device)
    combined_im = torch.zeros((3,H,W),device=device)
    combined_certainty = torch.zeros((H,W),device=device)
    for i, ref_path in enumerate(ref_paths):
        im1 = Image.open(im1_path).resize((W, H))
        im2 = Image.open(ref_path).resize((W, H))
        # Match
        warp, certainty = roma_model.match(im1_path, ref_path, device=device)
        # Sampling not needed, but can be done with model.sample(warp, certainty)
        x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
        x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)


        im2_transfer_rgb = F.grid_sample(
        x2[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
        )[0]
        print(combined_certainty.shape)
        mask = torch.gt(certainty[:,:W] , combined_certainty)
        mask_3 = torch.broadcast_to(mask, combined_im.shape) 
        combined_im[mask_3] = im2_transfer_rgb[mask_3]
        combined_certainty[mask] = (certainty[:,:W])[mask]
        warp_im = torch.cat((im2_transfer_rgb,x1),dim=2)
        certainty1 = torch.cat((certainty[:,:W],white_im[:,:W]),dim=-1)
        vis_im = certainty1 * warp_im + (1 - certainty1) * white_im
        tensor_to_pil(vis_im, unnormalize=False).save(f"compare_{i}.jpg")
    # combined_im /= combined_certainty
    # combined_certainty = torch.clamp(combined_certainty, max=1.0)
    combined_im = combined_certainty * combined_im + (1 - combined_certainty) * white_im[:,:W]
    gt = Image.open("/home/ccl/Datasets/NeRF-QA-Large-1/references/gt_horns/000.png").resize((W, H))
    gt = (torch.tensor(np.array(gt)) / 255).to(device).permute(2, 0, 1)
    tensor_to_pil(torch.cat((combined_im,gt),dim=2), unnormalize=False).save(f"combined.jpg")()