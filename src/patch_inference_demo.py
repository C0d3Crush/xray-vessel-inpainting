#!/usr/bin/env python3
"""
Patch inference that outputs 64x64 results for proper visualization
"""

import argparse, os, cv2, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from network.network_pro import Inpaint
from tqdm import tqdm
from utils import *
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np

parser = argparse.ArgumentParser(description="Patch inference with 64x64 output")
parser.add_argument('--ckpt', required=True, help='Path for the pretrained model')
parser.add_argument('--img_path', required=True, help='64x64 patch images directory')
parser.add_argument('--mask_path', required=True, help='64x64 mask patches directory')
parser.add_argument('--output_path', required=True, help='64x64 results directory')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--input_size', type=int, default=64, help='Patch size')
args = parser.parse_args()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

device = torch.device(args.device)
model = Inpaint(input_size=args.input_size)
model = load_checkpoint(args.ckpt, model, device)
model.eval().to(device)

maskfn = glob.glob(os.path.join(args.mask_path, '*.png'))
prog_bar = tqdm(maskfn, desc="Processing 64×64 patches")
psnr_scores = []

for mask_fn in prog_bar:
    fn = os.path.basename(mask_fn)
    img_fn = os.path.join(args.img_path, fn)
    
    if not os.path.exists(img_fn):
        continue
    
    # Load 64x64 patches
    gt_patch = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
    mask_patch = cv2.imread(mask_fn, cv2.IMREAD_GRAYSCALE)
    
    # Convert to tensors
    gt_tensor = torch.tensor((gt_patch.astype(np.float32) / 255.0) * 2.0 - 1.0)[None, None, :, :].to(device)
    mask_tensor = torch.tensor(mask_patch.astype(np.float32) / 255.0)[None, None, :, :].to(device)
    
    # Run inference
    with torch.no_grad():
        result = model(gt_tensor, mask_tensor)
    
    # Convert back to image
    result = torch.clamp(result, -1.0, 1.0) * 0.5 + 0.5
    result_np = result[0, 0].cpu().detach().numpy() * 255.0
    
    # Apply inpainting constraint
    mask_binary = (mask_patch > 127).astype(np.float32)
    final_result = result_np * mask_binary + gt_patch.astype(np.float32) * (1 - mask_binary)
    
    # Calculate PSNR
    psnr = calculate_psnr(gt_patch, final_result.astype(np.uint8))
    psnr_scores.append(psnr)
    
    # Save 64x64 result
    cv2.imwrite(os.path.join(args.output_path, fn), final_result.astype(np.uint8))
    prog_bar.set_postfix({"PSNR": f"{psnr:.2f}"})

avg_psnr = np.mean(psnr_scores) if psnr_scores else 0
print(f"\n✓ Patch inference complete!")
print(f"  Processed: {len(psnr_scores)} patches (64×64)")
print(f"  Average PSNR: {avg_psnr:.2f} dB")
print(f"  Results: {args.output_path}")