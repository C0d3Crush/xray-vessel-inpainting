#!/usr/bin/env python3
"""
Medical-aware hyperparameter optimization for foreground-biased patch sampling.
Tests different combinations with medical imaging constraints for optimal inpainting quality.
"""

import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import itertools
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def evaluate_medical_quality(metrics):
    """
    Medical-aware evaluation of training results.
    
    Args:
        metrics: Dictionary with training metrics
        
    Returns:
        quality_score: Float score (higher is better), -inf for invalid results
    """
    psnr = metrics.get('val_psnr', 0)
    ssim = metrics.get('val_ssim', 0)
    kl_div = metrics.get('val_kl_divergence', float('inf'))
    train_loss = metrics.get('train_loss', float('inf'))
    val_loss = metrics.get('val_loss', float('inf'))
    
    # Medical Reality Constraints
    if psnr > 70:
        print(f"   ❌ REJECTED: Unrealistic PSNR ({psnr:.1f} dB) - likely overfitting")
        return -float('inf')
    
    if train_loss < 0.05:
        print(f"   ❌ REJECTED: Trivial learning (loss={train_loss:.4f}) - empty patch problem")
        return -float('inf')
    
    if val_loss > train_loss * 3:
        print(f"   ❌ REJECTED: Severe overfitting (val_loss/train_loss = {val_loss/train_loss:.2f})")
        return -float('inf')
    
    # Quality Scoring (realistic PSNR range: 30-50 dB)
    if 30 <= psnr <= 50:
        psnr_score = 1.0 - abs(psnr - 40) / 10  # Peak at 40 dB
    elif 25 <= psnr < 30:
        psnr_score = 0.5 * (psnr - 25) / 5     # Gradual penalty below 30
    elif 50 < psnr <= 60:
        psnr_score = 0.7 * (60 - psnr) / 10    # Penalty for too high
    else:
        psnr_score = 0.1  # Very low score for unrealistic values
    
    # SSIM scoring (higher is better, but cap at 0.95 to avoid perfect scores)
    ssim_score = min(ssim, 0.95)
    
    # KL divergence scoring (lower is better)
    kl_score = np.exp(-kl_div) if kl_div < 10 else 0.01
    
    # Loss convergence scoring
    loss_ratio = val_loss / train_loss if train_loss > 0 else 1.0
    convergence_score = 1.0 - min(abs(loss_ratio - 1.0), 1.0)  # Peak when val_loss ≈ train_loss
    
    # Weighted combination (medical realism is most important)
    quality_score = (psnr_score * 0.4 + 
                    ssim_score * 0.3 + 
                    kl_score * 0.15 + 
                    convergence_score * 0.15)
    
    print(f"   📊 Quality Score: {quality_score:.3f} (PSNR: {psnr:.1f}, SSIM: {ssim:.3f}, KL: {kl_div:.3f})")
    return quality_score

def is_memory_feasible(config, device):
    """Check if configuration is memory-feasible."""
    total_patches = config['patches_per_image'] * config['batch_size']
    
    if device == 'cpu':
        return total_patches <= 128  # CPU memory limit
    elif device == 'cuda':
        return total_patches <= 512  # GPU memory limit (conservative)
    else:
        return total_patches <= 64   # Conservative fallback

def run_training_experiment(config, base_args):
    """Run a single training experiment with given hyperparameters."""
    
    # Create experiment name
    exp_name = f"fg{config['foreground_prob']}_ppi{config['patches_per_image']}_bs{config['batch_size']}_lr{config['lr']:.0e}"
    output_dir = f"experiments/{exp_name}"
    
    # Build command
    cmd = [
        "python", "src/train.py",
        "--smoke_test",
        "--smoke_size", str(base_args['smoke_size']),
        "--epochs", str(config['epochs']),
        "--patch_mode",
        "--input_size", str(base_args['input_size']),
        "--patches_per_image", str(config['patches_per_image']),
        "--batch_size", str(config['batch_size']),
        "--device", base_args['device'],
        "--train_img", base_args['train_img'],
        "--train_mask", base_args['train_mask'],
        "--val_img", base_args['val_img'],
        "--val_ann", base_args['val_ann'],
        "--output_dir", output_dir,
        "--foreground_prob", str(config['foreground_prob']),
        "--lr", str(config['lr'])
    ]
    
    print(f"\n🧪 Running experiment: {exp_name}")
    print(f"   Command: {' '.join(cmd[-8:])}")  # Show key params
    
    try:
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5min timeout
        
        if result.returncode != 0:
            print(f"❌ Experiment {exp_name} failed:")
            print(result.stderr[-500:])  # Last 500 chars of error
            return None
            
        # Parse results from training log
        log_path = Path(output_dir) / "training_log.csv"
        if log_path.exists():
            df = pd.read_csv(log_path)
            final_metrics = df.iloc[-1].to_dict()
            final_metrics['experiment'] = exp_name
            final_metrics['config'] = config
            
            # Medical-aware evaluation
            quality_score = evaluate_medical_quality(final_metrics)
            final_metrics['quality_score'] = quality_score
            
            if quality_score > -float('inf'):
                print(f"✅ {exp_name}: ACCEPTED")
                return final_metrics
            else:
                print(f"❌ {exp_name}: REJECTED (medical constraints)")
                return None
        else:
            print(f"❌ No log found for {exp_name}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment {exp_name} timed out")
        return None
    except Exception as e:
        print(f"❌ Experiment {exp_name} error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for CMT training")
    parser.add_argument("--smoke_size", type=int, default=25, help="Smoke test dataset size")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer combinations")
    
    # Data paths
    parser.add_argument("--train_img", default="data/smoke_bg_img", help="Training images")
    parser.add_argument("--train_mask", default="data/smoke_bg_mask", help="Training masks")
    parser.add_argument("--val_img", default="data/arcade/syntax/val/images", help="Validation images")
    parser.add_argument("--val_ann", default="data/arcade/syntax/val/annotations/val.json", help="Validation annotations")
    
    args = parser.parse_args()
    
    # Base arguments
    base_args = {
        'smoke_size': args.smoke_size,
        'input_size': 64,
        'device': args.device,
        'train_img': args.train_img,
        'train_mask': args.train_mask,
        'val_img': args.val_img,
        'val_ann': args.val_ann
    }
    
    # Hyperparameter grid
    if args.quick:
        # Quick test - fewer combinations
        param_grid = {
            'foreground_prob': [0.5, 0.75, 0.9],
            'patches_per_image': [4, 8],
            'batch_size': [8, 16],
            'lr': [5e-5, 1e-4],
            'epochs': [3]  # Quick epochs for testing
        }
    else:
        # Full grid search
        param_grid = {
            'foreground_prob': [0.5, 0.75, 0.8, 0.9, 0.95],
            'patches_per_image': [4, 8, 16],
            'batch_size': [8, 16, 32],
            'lr': [5e-5, 1e-4, 2e-4],
            'epochs': [5]  # Moderate epochs for optimization
        }
    
    # Create experiments directory
    Path("experiments").mkdir(exist_ok=True)
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    print(f"🚀 Starting hyperparameter optimization")
    print(f"   Total combinations: {len(combinations)}")
    print(f"   Device: {args.device}")
    print(f"   Mode: {'Quick' if args.quick else 'Full'}")
    
    # Run experiments
    results = []
    for i, config in enumerate(combinations, 1):
        print(f"\n📊 Progress: {i}/{len(combinations)}")
        
        # Memory feasibility check
        if not is_memory_feasible(config, args.device):
            total_patches = config['patches_per_image'] * config['batch_size']
            print(f"⚠️  Skipping memory-intensive config: {total_patches} patches (device: {args.device})")
            continue
            
        result = run_training_experiment(config, base_args)
        if result:
            results.append(result)
    
    # Analyze results
    if not results:
        print("❌ No successful experiments!")
        return
    
    df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiments/hyperparameter_results_{timestamp}.csv"
    df.to_csv(results_file, index=False)
    
    print(f"\n🎯 Hyperparameter Optimization Results")
    print(f"   Saved to: {results_file}")
    print(f"   Successful experiments: {len(results)}")
    
    # Best configurations by medical quality score
    print(f"\n🏆 TOP 5 CONFIGURATIONS (by Medical Quality Score):")
    print("=" * 90)
    
    # Sort by quality score (medical-aware evaluation)
    df_sorted = df.sort_values('quality_score', ascending=False)
    
    for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
        config = row['config'] if isinstance(row['config'], dict) else eval(row['config'])
        print(f"{i}. {row['experiment']}")
        print(f"   Quality Score: {row['quality_score']:.3f} | PSNR: {row['val_psnr']:.1f} dB | SSIM: {row['val_ssim']:.4f}")
        print(f"   Train Loss: {row['train_loss']:.4f} | Val Loss: {row['val_loss']:.4f} | KL Div: {row.get('val_kl_divergence', 'N/A')}")
        print(f"   Config: fg_prob={config['foreground_prob']}, ppi={config['patches_per_image']}, bs={config['batch_size']}, lr={config['lr']:.0e}")
        print()
    
    # Medical insights
    print("🏥 MEDICAL QUALITY INSIGHTS:")
    print("-" * 50)
    realistic_count = sum(1 for _, row in df.iterrows() if 30 <= row['val_psnr'] <= 50)
    overfitting_count = sum(1 for _, row in df.iterrows() if row['val_psnr'] > 70)
    trivial_count = sum(1 for _, row in df.iterrows() if row['train_loss'] < 0.05)
    
    print(f"Realistic PSNR (30-50 dB): {realistic_count}/{len(df)} ({realistic_count/len(df)*100:.1f}%)")
    print(f"Overfitting cases (PSNR > 70): {overfitting_count}/{len(df)} ({overfitting_count/len(df)*100:.1f}%)")
    print(f"Trivial learning (loss < 0.05): {trivial_count}/{len(df)} ({trivial_count/len(df)*100:.1f}%)")
    print()
    
    # Analysis by parameter
    print("📈 PARAMETER ANALYSIS:")
    print("-" * 40)
    
    for param in ['foreground_prob', 'patches_per_image', 'batch_size', 'lr']:
        if param in df.columns:
            continue
            
        # Extract parameter from config
        param_values = []
        quality_values = []
        
        for _, row in df.iterrows():
            config = row['config'] if isinstance(row['config'], dict) else eval(row['config'])
            param_values.append(config[param])
            quality_values.append(row['quality_score'])
        
        param_df = pd.DataFrame({param: param_values, 'quality': quality_values})
        avg_by_param = param_df.groupby(param)['quality'].mean().sort_values(ascending=False)
        
        print(f"{param.upper()}:")
        for val, quality in avg_by_param.head(3).items():
            print(f"  {val}: {quality:.3f} quality score (avg)")
        print()
    
    # Recommendations
    best_config = eval(df_sorted.iloc[0]['config'])
    print("🎯 RECOMMENDED HYPERPARAMETERS:")
    print("-" * 40)
    print(f"--foreground_prob {best_config['foreground_prob']}")
    print(f"--patches_per_image {best_config['patches_per_image']}")
    print(f"--batch_size {best_config['batch_size']}")
    print(f"--lr {best_config['lr']:.0e}")
    print(f"Expected PSNR: {df_sorted.iloc[0]['val_psnr']:.1f} dB")

if __name__ == "__main__":
    main()