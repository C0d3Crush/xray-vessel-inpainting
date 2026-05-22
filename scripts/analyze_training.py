#!/usr/bin/env python3
"""
Analyze training logs to detect unrealistic metrics and training issues.
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

def analyze_training_logs(log_path, analysis_path):
    """Analyze training logs and provide insights."""
    
    print("🔍 Training Analysis Report")
    print("=" * 50)
    
    # Load main training log
    try:
        df = pd.read_csv(log_path)
        print(f"📊 Loaded {len(df)} epochs from {log_path}")
    except FileNotFoundError:
        print(f"❌ Training log not found: {log_path}")
        return
    
    # Load analysis log if available
    analysis_df = None
    if Path(analysis_path).exists():
        try:
            analysis_df = pd.read_csv(analysis_path)
            print(f"📊 Loaded analysis data from {analysis_path}")
        except Exception as e:
            print(f"⚠️  Could not load analysis log: {e}")
    
    print("\n🎯 PSNR Analysis:")
    print("-" * 30)
    
    # PSNR analysis
    max_psnr = df['val_psnr'].max()
    min_psnr = df['val_psnr'].min()
    first_epoch_psnr = df['val_psnr'].iloc[0]
    
    print(f"First epoch PSNR: {first_epoch_psnr:.2f} dB")
    print(f"Max PSNR: {max_psnr:.2f} dB")
    print(f"Min PSNR: {min_psnr:.2f} dB")
    
    # PSNR realism check
    if first_epoch_psnr > 70:
        print("🚨 ALERT: First epoch PSNR > 70 dB - Task likely too easy!")
        print("   Possible causes:")
        print("   - Model copying input instead of inpainting")
        print("   - Masks too small or sparse")
        print("   - Validation on wrong data distribution")
    elif first_epoch_psnr > 50:
        print("⚠️  WARNING: First epoch PSNR > 50 dB - Suspiciously high")
    elif 30 <= first_epoch_psnr <= 45:
        print("✅ GOOD: First epoch PSNR in realistic range (30-45 dB)")
    else:
        print(f"📊 INFO: First epoch PSNR = {first_epoch_psnr:.2f} dB")
    
    print("\n🔄 Learning Pattern Analysis:")
    print("-" * 30)
    
    # Learning pattern analysis
    if len(df) >= 3:
        loss_changes = []
        for i in range(1, min(4, len(df))):
            change = df['train_loss'].iloc[i-1] - df['train_loss'].iloc[i]
            loss_changes.append(change)
            print(f"Epoch {i} → {i+1}: Loss change = {change:.6f}")
        
        avg_change = sum(loss_changes) / len(loss_changes)
        
        if avg_change < 0.001:
            print("🚨 ALERT: Loss barely changing - Model may have found trivial solution!")
        elif avg_change > 0.05:
            print("🚨 ALERT: Loss changing too rapidly - Check learning rate/data")
        elif avg_change > 0.005:
            print("✅ GOOD: Healthy learning rate")
        else:
            print("📊 INFO: Gradual learning")
    
    print("\n📈 Metric Progression:")
    print("-" * 30)
    
    # Show progression
    for i, row in df.head(5).iterrows():
        epoch = row['epoch']
        loss = row['train_loss']
        psnr = row['val_psnr']
        ssim = row['val_ssim']
        print(f"Epoch {epoch:2d}: Loss={loss:.4f} | PSNR={psnr:5.2f} dB | SSIM={ssim:.4f}")
    
    if len(df) > 5:
        print("...")
    
    # Analysis log insights
    if analysis_df is not None:
        print("\n🧠 Detailed Analysis:")
        print("-" * 30)
        
        realistic_count = (analysis_df['psnr_realistic'] == 'realistic').sum()
        too_high_count = (analysis_df['psnr_realistic'] == 'too_high').sum()
        good_learning_count = (analysis_df['learning_pattern'] == 'good_learning').sum()
        
        print(f"Epochs with realistic PSNR (30-45 dB): {realistic_count}/{len(analysis_df)}")
        print(f"Epochs with too high PSNR (>70 dB): {too_high_count}/{len(analysis_df)}")
        print(f"Epochs with good learning pattern: {good_learning_count}/{len(analysis_df)}")
        
        if too_high_count > len(analysis_df) // 2:
            print("🚨 MAJOR ISSUE: Most epochs have unrealistic PSNR!")
        elif realistic_count > len(analysis_df) // 2:
            print("✅ GOOD: Most epochs have realistic PSNR")
        
        # L1 vs SSIM loss analysis
        if 'l1_loss' in analysis_df.columns:
            avg_l1 = analysis_df['l1_loss'].mean()
            avg_ssim = analysis_df['ssim_loss'].mean()
            print(f"\nAverage L1 loss: {avg_l1:.6f}")
            print(f"Average SSIM loss: {avg_ssim:.6f}")
            
            l1_ratio = avg_l1 / (avg_l1 + avg_ssim)
            print(f"L1 dominance: {l1_ratio:.2%}")
            
            if l1_ratio > 0.95:
                print("⚠️  L1 loss dominates - SSIM loss may be too small")
            elif l1_ratio < 0.70:
                print("⚠️  SSIM loss dominates - L1 loss may be too small")
    
    print("\n🎯 Recommendations:")
    print("-" * 30)
    
    if first_epoch_psnr > 70:
        print("1. 🔍 Investigate why PSNR is so high:")
        print("   - Check mask sizes: Are they covering enough area?")
        print("   - Visualize outputs: Is model actually inpainting?")
        print("   - Check validation data: Same distribution as training?")
        print("2. 🎛️  Make task harder:")
        print("   - Increase background mask sizes")
        print("   - Use more complex shapes")
        print("   - Validate on vessel masks instead of background masks")
    elif 30 <= first_epoch_psnr <= 45:
        print("✅ Training appears healthy - continue monitoring")
    
    print("\n" + "=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Analyze training logs")
    parser.add_argument("--log", default="checkpoints_bg/training_log.csv", 
                       help="Path to training log CSV")
    parser.add_argument("--analysis", default="checkpoints_bg/training_analysis.csv",
                       help="Path to analysis log CSV")
    
    args = parser.parse_args()
    
    analyze_training_logs(args.log, args.analysis)

if __name__ == "__main__":
    main()