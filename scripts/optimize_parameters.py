#!/usr/bin/env python3
"""
Parameter optimization script for CMT inpainting model.
Tests different loss weight combinations to find optimal parameters.
"""

import os
import sys
import subprocess
import itertools
import pandas as pd
from pathlib import Path

def run_training(params, output_suffix):
    """Run training with specific parameters and return final validation metrics."""
    
    output_dir = f"checkpoints_opt_{output_suffix}"
    cmd = [
        "python", "src/train.py",
        "--train_img", "data/arcade/syntax/train/images",
        "--train_ann", "data/arcade/syntax/train/annotations/train.json", 
        "--val_img", "data/arcade/syntax/val/images",
        "--val_ann", "data/arcade/syntax/val/annotations/val.json",
        "--output_dir", output_dir,
        "--epochs", "10",  # Short training for parameter search
        "--batch_size", "4",
        "--input_size", "64",
        "--device", "cpu",
        "--ssim_weight", str(params['ssim_weight']),
        "--mask_weight", str(params['mask_weight']),
        "--valid_weight", str(params['valid_weight']),
        "--lr", str(params['lr'])
    ]
    
    print(f"\n🔄 Testing parameters: {params}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        if result.returncode != 0:
            print(f"❌ Training failed: {result.stderr}")
            return None
            
        # Read final metrics from training log
        log_path = os.path.join(output_dir, "training_log.csv")
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            final_metrics = df.iloc[-1].to_dict()
            final_metrics.update(params)  # Add parameter values
            print(f"✅ Final PSNR: {final_metrics['val_psnr']:.2f} dB, SSIM: {final_metrics['val_ssim']:.4f}")
            return final_metrics
        else:
            print(f"❌ No training log found: {log_path}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Training timed out after 10 minutes")
        return None
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None

def main():
    """Run parameter optimization sweep."""
    
    print("🔧 CMT Parameter Optimization")
    print("="*50)
    
    # Define parameter search space
    param_space = {
        'ssim_weight': [0.3, 0.5, 0.7],           # SSIM loss weight
        'mask_weight': [4.0, 6.0, 8.0],           # Masked region penalty
        'valid_weight': [0.5, 1.0, 2.0],          # Valid region penalty
        'lr': [5e-5, 1e-4, 2e-4]                  # Learning rate
    }
    
    print(f"📊 Parameter space:")
    for param, values in param_space.items():
        print(f"   {param}: {values}")
    
    # Generate all combinations
    param_names = list(param_space.keys())
    param_values = list(param_space.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"\n🚀 Testing {len(combinations)} parameter combinations...")
    
    results = []
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        suffix = f"run_{i:02d}"
        
        result = run_training(params, suffix)
        if result:
            results.append(result)
            
        print(f"Progress: {i+1}/{len(combinations)} ({(i+1)/len(combinations)*100:.1f}%)")
    
    # Analyze results
    if results:
        results_df = pd.DataFrame(results)
        
        # Save results
        results_path = "parameter_optimization_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n💾 Results saved to: {results_path}")
        
        # Find best parameters
        best_psnr = results_df.loc[results_df['val_psnr'].idxmax()]
        best_ssim = results_df.loc[results_df['val_ssim'].idxmax()]
        
        print("\n🏆 BEST PARAMETERS:")
        print(f"   Best PSNR: {best_psnr['val_psnr']:.2f} dB")
        print(f"      ssim_weight={best_psnr['ssim_weight']}, mask_weight={best_psnr['mask_weight']}")
        print(f"      valid_weight={best_psnr['valid_weight']}, lr={best_psnr['lr']}")
        
        print(f"\n   Best SSIM: {best_ssim['val_ssim']:.4f}")
        print(f"      ssim_weight={best_ssim['ssim_weight']}, mask_weight={best_ssim['mask_weight']}")
        print(f"      valid_weight={best_ssim['valid_weight']}, lr={best_ssim['lr']}")
        
        # Show top 3 combinations
        print(f"\n📈 TOP 3 BY PSNR:")
        top_psnr = results_df.nlargest(3, 'val_psnr')
        for idx, row in top_psnr.iterrows():
            print(f"   PSNR: {row['val_psnr']:.2f} dB | SSIM: {row['val_ssim']:.4f} | "
                  f"weights: [{row['ssim_weight']}, {row['mask_weight']}, {row['valid_weight']}] | "
                  f"lr: {row['lr']}")
                  
    else:
        print("❌ No successful parameter combinations found")

if __name__ == "__main__":
    main()