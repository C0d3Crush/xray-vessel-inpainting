#!/usr/bin/env python3
"""
Generate random background masks for training generalization.

Creates random geometric shapes (circles, rectangles, irregular blobs) in vessel-free regions
to teach the model to generate realistic background tissue, not just copy existing patterns.
"""

import argparse, logging, os, cv2, glob
import numpy as np

logger = logging.getLogger(__name__)
from scipy.ndimage import binary_dilation, binary_erosion
import random

def create_vessel_exclusion_mask(vessel_mask, safety_margin=5):
    """Create exclusion mask around vessels with safety margin"""
    # Convert vessel mask to binary
    vessel_binary = (vessel_mask > 127).astype(np.uint8)
    
    # Add safety margin around vessels
    struct = np.ones((2*safety_margin+1, 2*safety_margin+1), dtype=np.uint8)
    vessel_exclusion = binary_dilation(vessel_binary, structure=struct)
    
    return vessel_exclusion.astype(np.uint8)

def generate_random_circle(img_shape, exclusion_mask, min_radius=8, max_radius=25):
    """Generate random circle in vessel-free region"""
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Try multiple positions to find vessel-free location
    for _ in range(50):  # Max attempts
        radius = random.randint(min_radius, max_radius)
        center_x = random.randint(radius, w - radius)
        center_y = random.randint(radius, h - radius)
        
        # Check if circle area is vessel-free
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(temp_mask, (center_x, center_y), radius, 255, -1)
        
        # Check overlap with vessels
        overlap = np.sum((temp_mask > 0) & (exclusion_mask > 0))
        circle_area = np.sum(temp_mask > 0)
        
        if overlap / circle_area < 0.1:  # Less than 10% overlap allowed
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            return mask, True
    
    return mask, False

def generate_random_rectangle(img_shape, exclusion_mask, min_size=10, max_size=30):
    """Generate random rectangle in vessel-free region"""
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for _ in range(50):  # Max attempts
        rect_w = random.randint(min_size, max_size)
        rect_h = random.randint(min_size, max_size)
        x1 = random.randint(0, w - rect_w)
        y1 = random.randint(0, h - rect_h)
        x2 = x1 + rect_w
        y2 = y1 + rect_h
        
        # Check if rectangle area is vessel-free
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(temp_mask, (x1, y1), (x2, y2), 255, -1)
        
        overlap = np.sum((temp_mask > 0) & (exclusion_mask > 0))
        rect_area = np.sum(temp_mask > 0)
        
        if overlap / rect_area < 0.1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            return mask, True
    
    return mask, False

def generate_random_blob(img_shape, exclusion_mask, num_points=6, max_radius=20):
    """Generate irregular blob shape in vessel-free region"""
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for _ in range(50):  # Max attempts
        # Generate random center
        center_x = random.randint(max_radius, w - max_radius)
        center_y = random.randint(max_radius, h - max_radius)
        
        # Generate random polygon points around center
        points = []
        for i in range(num_points):
            angle = (2 * np.pi * i) / num_points + random.uniform(-0.3, 0.3)
            radius = random.uniform(max_radius * 0.5, max_radius)
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            points.append([x, y])
        
        # Draw filled polygon
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(temp_mask, [points_array], 255)
        
        overlap = np.sum((temp_mask > 0) & (exclusion_mask > 0))
        blob_area = np.sum(temp_mask > 0)
        
        if blob_area > 0 and overlap / blob_area < 0.1:
            cv2.fillPoly(mask, [points_array], 255)
            return mask, True
    
    return mask, False

def generate_background_training_mask(img_shape, vessel_mask, num_shapes=3, safety_margin=5):
    """Generate random background mask avoiding vessels"""
    h, w = img_shape
    
    # Create vessel exclusion zone
    exclusion_mask = create_vessel_exclusion_mask(vessel_mask, safety_margin)
    
    # Start with empty mask
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    successful_shapes = 0
    
    # Generate random shapes
    shape_generators = [
        generate_random_circle,
        generate_random_rectangle, 
        generate_random_blob
    ]
    
    for _ in range(num_shapes):
        # Randomly choose shape type
        generator = random.choice(shape_generators)
        shape_mask, success = generator(img_shape, exclusion_mask)
        
        if success:
            # Add to combined mask (union operation)
            combined_mask = cv2.bitwise_or(combined_mask, shape_mask)
            successful_shapes += 1
    
    return combined_mask, successful_shapes

def process_training_masks(input_img_dir, input_mask_dir, output_img_dir, output_mask_dir, 
                         num_variations=3, safety_margin=5):
    """Process all images to create background training masks"""
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    img_files = glob.glob(os.path.join(input_img_dir, "*.png"))
    total_generated = 0
    
    for img_file in img_files:
        filename = os.path.basename(img_file)
        base_name = os.path.splitext(filename)[0]
        vessel_mask_file = os.path.join(input_mask_dir, filename)
        
        if not os.path.exists(vessel_mask_file):
            continue
            
        # Load image and vessel mask
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        vessel_mask = cv2.imread(vessel_mask_file, cv2.IMREAD_GRAYSCALE)
        
        # Generate multiple variations per image
        for var_idx in range(num_variations):
            var_suffix = f"_bg_{var_idx:02d}"
            
            # Generate background mask avoiding vessels
            bg_mask, num_shapes = generate_background_training_mask(
                img.shape, vessel_mask, num_shapes=random.randint(2, 5), 
                safety_margin=safety_margin
            )
            
            if num_shapes > 0:  # Only save if we successfully generated shapes
                # Save image copy and background mask
                output_img_file = os.path.join(output_img_dir, f"{base_name}{var_suffix}.png")
                output_mask_file = os.path.join(output_mask_dir, f"{base_name}{var_suffix}.png")
                
                cv2.imwrite(output_img_file, img)
                cv2.imwrite(output_mask_file, bg_mask)
                total_generated += 1
                
                logger.debug(f"{base_name}{var_suffix}: {num_shapes} background shapes generated")

    logger.info(f"Generated {total_generated} background training samples")
    logger.info(f"Images: {output_img_dir} | Masks: {output_mask_dir} | Safety margin: {safety_margin}px")

def main():
    parser = argparse.ArgumentParser(description="Generate background training masks")
    parser.add_argument("--input-img", required=True, help="Input images directory")
    parser.add_argument("--input-mask", required=True, help="Vessel masks directory")
    parser.add_argument("--output-img", required=True, help="Output images directory") 
    parser.add_argument("--output-mask", required=True, help="Output background masks directory")
    parser.add_argument("--variations", type=int, default=3, help="Variations per image")
    parser.add_argument("--safety-margin", type=int, default=5, help="Safety margin around vessels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    logger.info(f"Generating background training masks: input={args.input_img}, masks={args.input_mask}, variations={args.variations}, margin={args.safety_margin}px")
    
    process_training_masks(
        args.input_img, args.input_mask,
        args.output_img, args.output_mask,
        args.variations, args.safety_margin
    )

if __name__ == "__main__":
    main()