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

    # Create figure with 3 subplots
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
