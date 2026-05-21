#!/usr/bin/env python3
"""
Create enhanced training comparison visualization in the style of training_comparison.png

Compares patch training vs resize training (or other training methods) side-by-side
with full-size image panels, clear labels, and professional layout.
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def load_image(path):
    """Load and convert image to grayscale numpy array"""
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert('L')
    return np.array(img)

def create_comparison_panel(original, mask, result, title, panel_size=(200, 200)):
    """Create a single comparison panel with original|mask|result"""
    if original is None or mask is None or result is None:
        # Create placeholder panel
        panel_width = panel_size[0] * 3 + 20  # 3 images + 2 gaps
        panel_height = panel_size[1] + 40  # image + title space
        panel = Image.new('RGB', (panel_width, panel_height), 'white')
        draw = ImageDraw.Draw(panel)
        draw.text((10, 10), f"{title} - Images not found", fill='red')
        return panel
    
    # Resize images to panel size
    orig_img = Image.fromarray(original).resize(panel_size, Image.Resampling.LANCZOS)
    mask_img = Image.fromarray(mask).resize(panel_size, Image.Resampling.LANCZOS)
    result_img = Image.fromarray(result).resize(panel_size, Image.Resampling.LANCZOS)
    
    # Create panel
    panel_width = panel_size[0] * 3 + 20  # 3 images + 2 gaps
    panel_height = panel_size[1] + 40  # image + title space
    panel = Image.new('RGB', (panel_width, panel_height), 'white')
    
    # Paste images
    panel.paste(orig_img, (0, 30))
    panel.paste(mask_img, (panel_size[0] + 10, 30))
    panel.paste(result_img, (panel_size[0] * 2 + 20, 30))
    
    # Add title and labels
    draw = ImageDraw.Draw(panel)
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
    
    # Title
    bbox = draw.textbbox((0, 0), title, font=font_title)
    text_width = bbox[2] - bbox[0]
    x_center = (panel_width - text_width) // 2
    draw.text((x_center, 5), title, fill='black', font=font_title)
    
    # Labels
    draw.text((panel_size[0]//2 - 25, panel_height - 15), "Original", fill='black', font=font_label)
    draw.text((panel_size[0] + panel_size[0]//2 - 15, panel_height - 15), "Mask", fill='black', font=font_label)
    draw.text((panel_size[0]*2 + panel_size[0]//2 - 20, panel_height - 15), "Result", fill='black', font=font_label)
    
    return panel

def create_training_comparison(
    patch_img_dir, patch_mask_dir, patch_result_dir,
    resize_img_dir=None, resize_mask_dir=None, resize_result_dir=None,
    output_path="training_comparison_new.png",
    image_ids=None,
    comparison_title="Current Model Inpainting Results"
):
    """
    Create training comparison visualization
    
    Args:
        patch_*_dir: Directories for patch training results
        resize_*_dir: Directories for resize training results (optional)
        output_path: Where to save the comparison
        image_ids: List of image IDs to compare (e.g., ['124', '66'])
        comparison_title: Main title for the comparison
    """
    
    # Get available images
    if image_ids is None:
        # Auto-detect from patch results
        patch_images = [f.replace('.png', '') for f in os.listdir(patch_result_dir) 
                       if f.endswith('.png')]
        image_ids = sorted(patch_images)[:4]  # Take first 4
    
    # Determine layout
    have_resize = (resize_img_dir is not None and resize_mask_dir is not None 
                  and resize_result_dir is not None)
    
    if have_resize:
        # 2x2 comparison layout
        rows, cols = 2, 2
        fig_width = 1400
        fig_height = 800
        panel_size = (180, 180)
    else:
        # Single column layout for patch results only
        rows, cols = len(image_ids), 1
        fig_width = 700
        fig_height = 300 * len(image_ids)
        panel_size = (200, 200)
    
    # Create main figure
    fig = Image.new('RGB', (fig_width, fig_height), 'white')
    draw = ImageDraw.Draw(fig)
    
    # Add main title
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        title_font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), comparison_title, font=title_font)
    title_width = bbox[2] - bbox[0]
    title_x = (fig_width - title_width) // 2
    draw.text((title_x, 20), comparison_title, fill='black', font=title_font)
    
    # Create panels
    y_offset = 80
    
    for i, img_id in enumerate(image_ids):
        # Load patch training results
        patch_orig = load_image(os.path.join(patch_img_dir, f"{img_id}.png"))
        patch_mask = load_image(os.path.join(patch_mask_dir, f"{img_id}.png"))
        patch_result = load_image(os.path.join(patch_result_dir, f"{img_id}.png"))
        
        # Create patch panel
        patch_title = f"Image {img_id} - Patch Training\n(Original | Mask | Result)"
        patch_panel = create_comparison_panel(
            patch_orig, patch_mask, patch_result, patch_title, panel_size
        )
        
        if have_resize:
            # Load resize training results
            resize_orig = load_image(os.path.join(resize_img_dir, f"{img_id}.png"))
            resize_mask = load_image(os.path.join(resize_mask_dir, f"{img_id}.png"))
            resize_result = load_image(os.path.join(resize_result_dir, f"{img_id}.png"))
            
            # Create resize panel
            resize_title = f"Image {img_id} - Resize Training\n(Original | Mask | Result)"
            resize_panel = create_comparison_panel(
                resize_orig, resize_mask, resize_result, resize_title, panel_size
            )
            
            # Position panels side by side
            row = i // 2
            col = i % 2
            
            if col == 0:  # Left column - patch training
                x_pos = 50
            else:  # Right column - resize training  
                x_pos = fig_width // 2 + 50
            
            y_pos = y_offset + row * (panel_size[1] + 100)
            
            if col == 0:
                fig.paste(patch_panel, (x_pos, y_pos))
            else:
                fig.paste(resize_panel, (x_pos, y_pos))
        else:
            # Single column layout
            x_pos = (fig_width - patch_panel.width) // 2
            y_pos = y_offset + i * (panel_size[1] + 80)
            fig.paste(patch_panel, (x_pos, y_pos))
    
    # Save comparison
    fig.save(output_path, quality=95)
    print(f"✓ Training comparison saved: {output_path}")
    print(f"  Format: {fig.width}×{fig.height} pixels")
    print(f"  Images: {len(image_ids)} samples")
    
    # Clean up old comparison files if output is in samples directory
    if "samples" in str(output_path):
        old_comparisons_dir = Path(output_path).parent / "comparisons"
        if old_comparisons_dir.exists():
            old_files = list(old_comparisons_dir.glob("*.png"))
            if old_files:
                for f in old_files:
                    f.unlink()
                print(f"  Removed {len(old_files)} old comparison files from {old_comparisons_dir}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Create training comparison visualization")
    parser.add_argument("--patch-img", required=True, help="Patch training input images directory")
    parser.add_argument("--patch-mask", required=True, help="Patch training masks directory") 
    parser.add_argument("--patch-result", required=True, help="Patch training results directory")
    parser.add_argument("--resize-img", help="Resize training input images directory")
    parser.add_argument("--resize-mask", help="Resize training masks directory")
    parser.add_argument("--resize-result", help="Resize training results directory") 
    parser.add_argument("--output", default="training_comparison_new.png", help="Output comparison image")
    parser.add_argument("--images", nargs="+", help="Specific image IDs to compare")
    parser.add_argument("--title", default="Model Training Comparison", help="Main comparison title")
    
    args = parser.parse_args()
    
    create_training_comparison(
        patch_img_dir=args.patch_img,
        patch_mask_dir=args.patch_mask, 
        patch_result_dir=args.patch_result,
        resize_img_dir=args.resize_img,
        resize_mask_dir=args.resize_mask,
        resize_result_dir=args.resize_result,
        output_path=args.output,
        image_ids=args.images,
        comparison_title=args.title
    )

if __name__ == "__main__":
    main()