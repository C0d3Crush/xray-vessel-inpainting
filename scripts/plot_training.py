#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot training metrics from training_log.csv

Usage:
    python scripts/plot_training.py
    python scripts/plot_training.py checkpoints/training_log.csv --output training_plot.png
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_training_log(csv_path, output_path=None):
    """Plot training metrics from CSV log."""
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Check which metrics are available
    has_extended_metrics = 'val_wasserstein' in df.columns and 'val_rmse' in df.columns
    
    if has_extended_metrics:
        # Create figure with 5 subplots (2 rows)
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        axes = axes.flatten()  # Make indexing easier
        
        # Plot 1: Training Loss
        axes[0].plot(df['epoch'], df['train_loss'], 'b-', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Training Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Validation PSNR
        axes[1].plot(df['epoch'], df['val_psnr'], 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('PSNR (dB)', fontsize=12)
        axes[1].set_title('Validation PSNR', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Validation SSIM
        axes[2].plot(df['epoch'], df['val_ssim'], 'r-', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('SSIM', fontsize=12)
        axes[2].set_title('Validation SSIM', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Wasserstein Distance
        axes[3].plot(df['epoch'], df['val_wasserstein'], 'm-', linewidth=2)
        axes[3].set_xlabel('Epoch', fontsize=12)
        axes[3].set_ylabel('Wasserstein Distance', fontsize=12)
        axes[3].set_title('Validation Wasserstein Distance', fontsize=14, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        
        # Plot 5: RMSE
        axes[4].plot(df['epoch'], df['val_rmse'], 'c-', linewidth=2)
        axes[4].set_xlabel('Epoch', fontsize=12)
        axes[4].set_ylabel('RMSE', fontsize=12)
        axes[4].set_title('Validation RMSE', fontsize=14, fontweight='bold')
        axes[4].grid(True, alpha=0.3)
        
        # Hide the empty 6th subplot
        axes[5].set_visible(False)
        
    else:
        # Create figure with 3 subplots (legacy format)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot 1: Training Loss
        axes[0].plot(df['epoch'], df['train_loss'], 'b-', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Training Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Validation PSNR
        axes[1].plot(df['epoch'], df['val_psnr'], 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('PSNR (dB)', fontsize=12)
        axes[1].set_title('Validation PSNR', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Validation SSIM
        axes[2].plot(df['epoch'], df['val_ssim'], 'r-', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('SSIM', fontsize=12)
        axes[2].set_title('Validation SSIM', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from CSV")
    parser.add_argument('csv', nargs='?', default='checkpoints/training_log.csv',
                        help='Path to training_log.csv (default: checkpoints/training_log.csv)')
    parser.add_argument('--output', '-o', default='training_plot.png',
                        help='Output image path (default: training_plot.png)')
    parser.add_argument('--show', action='store_true',
                        help='Show plot instead of saving')
    args = parser.parse_args()

    # Check file exists
    if not Path(args.csv).exists():
        print(f"Error: {args.csv} not found")
        return

    # Plot
    plot_training_log(args.csv, None if args.show else args.output)


if __name__ == '__main__':
    main()
