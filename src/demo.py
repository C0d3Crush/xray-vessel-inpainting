# -*- coding: utf-8 -*-
import argparse, os, cv2, glob
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from network.network_pro import Inpaint
from tqdm import tqdm
from utils import *
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser(
    description="Official Pytorch Code for K. Ko and C.-S. Kim, Continuously Masked Transformer for Image Inpainting, ICCV 2023",
    usage='use "%(prog)s --help" for more information',
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument('--ckpt',        required=True,          help='Path for the pretrained model')
parser.add_argument('--img_path',    default="./samples/test_img",  help='Path for directory of images. File names must match mask names.')
parser.add_argument('--mask_path',   default="./samples/test_mask", help='Path for directory of masks.')
parser.add_argument('--output_path', default="./samples/results",   help='Path for saving inpainted images')
parser.add_argument('--device',      type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--input_size',  type=int, default=256,   help='Must match the input_size used during training (default: 256)')
args = parser.parse_args()

assert os.path.exists(args.img_path),  "Please check image path"
assert os.path.exists(args.mask_path), "Please check mask path"
if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

device   = torch.device(args.device)
proposed = Inpaint(input_size=args.input_size)
proposed = load_checkpoint(args.ckpt, proposed, device)
proposed.eval().to(device)

maskfn   = glob.glob(os.path.join(args.mask_path, '*.*'))
prog_bar = tqdm(maskfn)
avg      = 0.0

for step, mask_fn in enumerate(prog_bar):
    fn      = os.path.basename(mask_fn)
    gt_gray = cv2.imread(os.path.join(args.img_path, fn), cv2.IMREAD_GRAYSCALE)
    assert gt_gray is not None, "Could not read image: {}".format(os.path.join(args.img_path, fn))
    
    # Store original dimensions for output
    original_h, original_w = gt_gray.shape
    original_mask_gray = cv2.imread(mask_fn, cv2.IMREAD_GRAYSCALE)
    assert original_mask_gray is not None, "Could not read mask: {}".format(mask_fn)

    gt_  = (gt_gray.astype(np.float32) / 255.0) * 2.0 - 1.0
    mask = original_mask_gray.astype(np.float32) / 255.0

    gt   = torch.Tensor(gt_)[None, None, :, :].to(device, dtype=torch.float32)
    mask = torch.Tensor(mask)[None, None, :, :].to(device, dtype=torch.float32)

    # Resize to model input size for inference
    if gt.shape[-1] != args.input_size or gt.shape[-2] != args.input_size:
        gt_resized   = F.interpolate(gt,   size=(args.input_size, args.input_size), mode='bilinear', align_corners=False)
        mask_resized = F.interpolate(mask, size=(args.input_size, args.input_size), mode='nearest')
    else:
        gt_resized = gt
        mask_resized = mask

    with torch.no_grad():
        out_pro = proposed(gt_resized, mask_resized)

    out_pro = torch.clip(out_pro, -1.0, 1.0) * 0.5 + 0.5
    out_pro = out_pro[0, 0].cpu().detach().numpy() * 255.0
    
    # Resize result back to original dimensions
    if original_h != args.input_size or original_w != args.input_size:
        out_pro_full = cv2.resize(out_pro, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        mask_binary_full = (original_mask_gray > 127).astype(np.float32)
    else:
        out_pro_full = out_pro
        mask_binary_full = (original_mask_gray > 127).astype(np.float32)
    
    # The model already applies inpainting constraint internally
    # Let's use the model output directly first to debug
    final_result = out_pro_full

    score      = psnr(final_result, gt_gray.astype(np.float32))
    save_path_ = os.path.join(args.output_path, '{}'.format(fn))
    cv2.imwrite(save_path_, np.clip(final_result, 0, 255).astype(np.uint8))

    avg += score
    prog_bar.set_description("PSNR {}".format(avg / (step + 1)))
