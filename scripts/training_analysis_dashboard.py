#!/usr/bin/env python3
"""
Advanced training analysis dashboard for medical image inpainting.
Provides comprehensive analysis of training metrics with medical-aware insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_training_convergence(df):
    """Analyze training and validation loss convergence."""
    print("🔄 TRAINING CONVERGENCE ANALYSIS:")
    print("-" * 50)
    
    if 'val_loss' not in df.columns:
        print("❌ Validation loss not available in this log")
        return
    
    # Calculate convergence metrics
    final_train_loss = df['train_loss'].iloc[-1]
    final_val_loss = df['val_loss'].iloc[-1]
    loss_ratio = final_val_loss / final_train_loss
    
    # Loss trend analysis
    if len(df) >= 5:
        train_trend = np.polyfit(range(len(df)), df['train_loss'], 1)[0]
        val_trend = np.polyfit(range(len(df)), df['val_loss'], 1)[0]
        
        print(f"Final train loss: {final_train_loss:.4f}")
        print(f"Final val loss: {final_val_loss:.4f}")
        print(f"Val/Train ratio: {loss_ratio:.2f}")
        print(f"Train loss trend: {'↓' if train_trend < 0 else '↑'} {abs(train_trend):.6f}/epoch")
        print(f"Val loss trend: {'↓' if val_trend < 0 else '↑'} {abs(val_trend):.6f}/epoch")
        
        # Convergence assessment
        if loss_ratio > 2.0:
            print("⚠️  WARNING: Possible overfitting (val_loss >> train_loss)")
        elif loss_ratio < 1.1:
            print("✅ GOOD: Well-converged training")
        elif abs(train_trend) < 1e-5:
            print("⚠️  WARNING: Training has plateaued")
        else:
            print("📊 INFO: Normal training progression")
    
    print()

def analyze_medical_realism(df):
    """Analyze metrics from medical imaging perspective."""
    print("🏥 MEDICAL REALISM ANALYSIS:")
    print("-" * 50)
    
    psnr_values = df['val_psnr']
    ssim_values = df['val_ssim']
    
    # PSNR analysis
    realistic_epochs = sum(1 for psnr in psnr_values if 30 <= psnr <= 50)
    overfitting_epochs = sum(1 for psnr in psnr_values if psnr > 70)
    
    print(f"PSNR Analysis:")
    print(f"  Realistic range (30-50 dB): {realistic_epochs}/{len(df)} epochs ({realistic_epochs/len(df)*100:.1f}%)")
    print(f"  Overfitting (>70 dB): {overfitting_epochs}/{len(df)} epochs ({overfitting_epochs/len(df)*100:.1f}%)")
    print(f"  Final PSNR: {psnr_values.iloc[-1]:.1f} dB")
    print(f"  Max PSNR: {psnr_values.max():.1f} dB (epoch {psnr_values.idxmax() + 1})")
    
    # SSIM analysis
    print(f"\nSSIM Analysis:")
    print(f"  Final SSIM: {ssim_values.iloc[-1]:.4f}")
    print(f"  Max SSIM: {ssim_values.max():.4f} (epoch {ssim_values.idxmax() + 1})")
    
    # Medical quality assessment
    final_psnr = psnr_values.iloc[-1]
    final_ssim = ssim_values.iloc[-1]
    
    if 35 <= final_psnr <= 45 and final_ssim >= 0.85:
        print(f"\n✅ EXCELLENT: Medical-grade inpainting quality achieved")
    elif 30 <= final_psnr <= 50 and final_ssim >= 0.80:
        print(f"\n✅ GOOD: Clinically acceptable quality")
    elif final_psnr > 60:
        print(f"\n❌ SUSPICIOUS: Unrealistically high PSNR - check for overfitting")
    else:
        print(f"\n⚠️  MODERATE: Quality may need improvement")
    
    print()

def analyze_loss_components(df):
    """Analyze L1 and SSIM loss components."""
    print("🧮 LOSS COMPONENT ANALYSIS:")
    print("-" * 50)
    
    if 'val_l1_loss' not in df.columns or 'val_ssim_loss' not in df.columns:
        print("❌ Detailed loss components not available")
        return
    
    final_l1 = df['val_l1_loss'].iloc[-1]
    final_ssim = df['val_ssim_loss'].iloc[-1]
    l1_dominance = final_l1 / (final_l1 + final_ssim)
    
    print(f"Final L1 loss: {final_l1:.4f}")
    print(f"Final SSIM loss: {final_ssim:.4f}")
    print(f"L1 dominance: {l1_dominance:.1%}")
    
    if l1_dominance > 0.95:
        print("⚠️  L1 loss dominates - SSIM loss may be too small")
    elif l1_dominance < 0.70:
        print("⚠️  SSIM loss dominates - L1 loss may be too small")
    else:
        print("✅ Balanced loss components")
    
    print()

def analyze_advanced_metrics(df):
    """Analyze advanced metrics (KL divergence, Wasserstein)."""
    print("📊 ADVANCED METRICS ANALYSIS:")
    print("-" * 50)
    
    # KL divergence analysis
    if 'val_kl_divergence' in df.columns:
        kl_values = df['val_kl_divergence']
        print(f"KL Divergence:")
        print(f"  Final: {kl_values.iloc[-1]:.4f}")
        print(f"  Min: {kl_values.min():.4f} (epoch {kl_values.idxmin() + 1})")
        print(f"  Trend: {'↓' if kl_values.iloc[-1] < kl_values.iloc[0] else '↑'}")
        
        if kl_values.iloc[-1] < 0.1:
            print("  ✅ Excellent distribution matching")
        elif kl_values.iloc[-1] < 0.5:
            print("  ✅ Good distribution similarity")
        else:
            print("  ⚠️  High distribution divergence")
    
    # Wasserstein distance analysis
    if 'val_wasserstein' in df.columns:
        ws_values = df['val_wasserstein']
        print(f"\nWasserstein Distance:")
        print(f"  Final: {ws_values.iloc[-1]:.2f}")
        print(f"  Min: {ws_values.min():.2f} (epoch {ws_values.idxmin() + 1})")
        print(f"  Trend: {'↓' if ws_values.iloc[-1] < ws_values.iloc[0] else '↑'}")
    
    # RMSE analysis
    if 'val_rmse' in df.columns:
        rmse_values = df['val_rmse']
        print(f"\nRMSE:")
        print(f"  Final: {rmse_values.iloc[-1]:.2f}")
        print(f"  Min: {rmse_values.min():.2f} (epoch {rmse_values.idxmin() + 1})")
    
    print()

def create_training_plots(df, output_dir):
    """Create comprehensive training visualization plots."""
    print("📈 Creating training visualization plots...")
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Medical Image Inpainting Training Analysis', fontsize=16)
    
    # 1. Loss curves
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', alpha=0.7)
    if 'val_loss' in df.columns:
        ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', color='red', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. PSNR with medical range highlighting
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['val_psnr'], color='green', linewidth=2)
    ax2.axhspan(30, 50, alpha=0.2, color='green', label='Medical Range (30-50 dB)')
    ax2.axhspan(70, 100, alpha=0.2, color='red', label='Overfitting Zone (>70 dB)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('PSNR Evolution (Medical Perspective)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. SSIM progression
    ax3 = axes[0, 2]
    ax3.plot(df['epoch'], df['val_ssim'], color='purple', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('SSIM')
    ax3.set_title('Structural Similarity (SSIM)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Advanced metrics
    ax4 = axes[1, 0]
    if 'val_kl_divergence' in df.columns:
        ax4.plot(df['epoch'], df['val_kl_divergence'], label='KL Divergence', color='orange')
    if 'val_wasserstein' in df.columns:
        ax4_twin = ax4.twinx()
        ax4_twin.plot(df['epoch'], df['val_wasserstein'], label='Wasserstein', color='brown', linestyle='--')
        ax4_twin.set_ylabel('Wasserstein Distance')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('KL Divergence')
    ax4.set_title('Distribution Similarity Metrics')
    ax4.legend(loc='upper left')
    if 'val_wasserstein' in df.columns:
        ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 5. Loss components
    ax5 = axes[1, 1]
    if 'val_l1_loss' in df.columns and 'val_ssim_loss' in df.columns:
        ax5.plot(df['epoch'], df['val_l1_loss'], label='L1 Loss', color='blue')
        ax5.plot(df['epoch'], df['val_ssim_loss'], label='SSIM Loss', color='red')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss Component')
    ax5.set_title('L1 vs SSIM Loss Components')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Training health indicators
    ax6 = axes[1, 2]
    if 'val_loss' in df.columns:
        loss_ratio = df['val_loss'] / df['train_loss']
        ax6.plot(df['epoch'], loss_ratio, color='darkred', linewidth=2)
        ax6.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Convergence')
        ax6.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Val Loss / Train Loss')
        ax6.set_title('Overfitting Monitor')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    output_path = Path(output_dir) / "training_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Plots saved to: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Advanced training analysis dashboard")
    parser.add_argument("log_path", help="Path to training_log.csv")
    parser.add_argument("--output_dir", default=".", help="Output directory for plots")
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation")
    
    args = parser.parse_args()
    
    # Load training log
    try:
        df = pd.read_csv(args.log_path)
        print(f"📊 Loaded training log: {args.log_path}")
        print(f"   Epochs: {len(df)}")
        print(f"   Metrics: {', '.join(df.columns[1:])}")  # Skip 'epoch' column
        print()
    except FileNotFoundError:
        print(f"❌ Training log not found: {args.log_path}")
        return
    except Exception as e:
        print(f"❌ Error loading log: {e}")
        return
    
    # Run analyses
    analyze_training_convergence(df)
    analyze_medical_realism(df)
    analyze_loss_components(df)
    analyze_advanced_metrics(df)
    
    # Generate plots
    if not args.no_plots:
        create_training_plots(df, args.output_dir)
    
    # Summary recommendations
    print("🎯 TRAINING RECOMMENDATIONS:")
    print("-" * 50)
    
    final_psnr = df['val_psnr'].iloc[-1]
    final_ssim = df['val_ssim'].iloc[-1]
    
    if final_psnr > 60:
        print("1. 🔍 Investigate overfitting - PSNR too high")
        print("2. 📊 Check patch sampling - ensure meaningful mask coverage")
        print("3. 🎛️  Consider reducing foreground_prob or increasing dataset diversity")
    elif final_psnr < 25:
        print("1. 🎯 Improve model capacity or training duration")
        print("2. 📈 Check loss convergence - may need more epochs")
        print("3. 🔧 Tune hyperparameters (learning rate, loss weights)")
    elif 30 <= final_psnr <= 50:
        print("✅ Training appears successful - realistic medical imaging quality achieved!")
        if final_ssim >= 0.85:
            print("✅ Excellent structural preservation")
        print("💡 Consider testing on vessel inpainting task")
    
    print("\n🏁 Analysis complete!")

if __name__ == "__main__":
    main()