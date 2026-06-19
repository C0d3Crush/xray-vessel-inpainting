# -*- coding: utf-8 -*-
import os
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

from network.network_pro import Inpaint
from utils import load_checkpoint, save_checkpoint, rotate_checkpoints, wasserstein_distance_2d, calculate_kl_divergence
from dataset import ArcadeDataset, DatasetConfig
from losses import InpaintingLoss, ssim_fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(
    # Data paths
    train_img='data/arcade/syntax/train/images',
    train_ann='data/arcade/syntax/train/annotations/train.json',
    train_mask=None,
    val_img='data/arcade/syntax/val/images',
    val_ann='data/arcade/syntax/val/annotations/val.json',
    val_mask=None,

    # Training parameters
    epochs=100,
    batch_size=4,
    lr=1e-4,
    input_size=256,
    device='cpu',

    # Model parameters
    output_dir='checkpoints',
    ckpt=None,

    # Data augmentation
    random_masks=False,
    mask_padding=10,
    patches_per_image=4,
    foreground_prob=0.75,
    max_shapes=5,

    # Loss parameters
    ssim_weight=0.5,
    mask_weight=6.0,
    valid_weight=1.0,

    # Other parameters
    seed=42,
    smoke_test=False,
    smoke_size=2,
    num_workers=2,
    save_every=10,
    keep_checkpoints=3,

    # Advanced training modes
    vessel_safe_training=False,
    background_training=True,

    # Google Drive mirroring (Colab only)
    drive_dir=None,

    # Automatic Mixed Precision (CUDA only — ignored on CPU/MPS)
    amp=False,

    # Callback for real-time monitoring (notebook integration)
    epoch_callback=None
):
    """
    Train CMT inpainting model with given parameters.

    Args:
        epoch_callback: Optional function(epoch, metrics_dict) called after each epoch
                       for real-time monitoring in notebooks

    Returns:
        dict: Training results including best_val_psnr, log_path, final_metrics
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    set_seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)

    # ---- Datasets ----
    base_cfg = DatasetConfig(
        image_size=input_size,
        patches_per_image=patches_per_image,
        foreground_prob=foreground_prob,
        max_shapes=max_shapes,
        background_training=background_training,
        vessel_safe_training=vessel_safe_training,
    )
    train_cfg = DatasetConfig(
        **{**base_cfg.__dict__,
           'mask_dir': train_mask,
           'random_masks': random_masks,
           'mask_padding': mask_padding},
    )
    val_cfg = DatasetConfig(**{**base_cfg.__dict__, 'mask_dir': val_mask})

    train_dataset = ArcadeDataset(train_img, train_ann, train_cfg)
    val_dataset   = ArcadeDataset(val_img,   val_ann,   val_cfg)

    if train_mask:
        logger.info(f"Using precomputed train masks from: {train_mask}")
    elif random_masks:
        logger.info(f"Generating random masks around vessel regions (padding: {mask_padding}px)")
    elif vessel_safe_training:
        logger.warning("Generating vessel-safe background masks (SLOW - consider using --train_mask for speed)")
    else:
        logger.info("Generating train masks from COCO annotations")

    if smoke_test:
        train_dataset.image_ids = train_dataset.image_ids[:smoke_size]
        val_dataset.image_ids   = val_dataset.image_ids[:max(1, smoke_size // 2)]

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    base_train_imgs = len(train_dataset.image_ids)
    base_val_imgs = len(val_dataset.image_ids)
    logger.info(f"Patch mode enabled: {patches_per_image} patches per image")
    logger.info(f"Train: {base_train_imgs} images → {len(train_dataset)} patches | Val: {base_val_imgs} images → {len(val_dataset)} patches")

    # ---- Model ----
    model = Inpaint(input_size=input_size).to(device)

    if ckpt and os.path.exists(ckpt):
        model = load_checkpoint(ckpt, model, device)
        logger.info(f"Resumed from checkpoint: {ckpt}")

    # ---- Optimiser & Loss ----
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = InpaintingLoss(ssim_weight=ssim_weight,
                                mask_weight=mask_weight,
                                valid_weight=valid_weight).to(device)

    # ---- AMP setup ----
    use_amp = amp and device.type == 'cuda'
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        logger.info("AMP enabled: using float16 for forward pass")

    # ---- Training loop ----
    best_val_psnr = 0.0
    log_path = os.path.join(output_dir, 'training_log.csv')

    use_drive = drive_dir is not None and os.path.isdir(drive_dir)
    if use_drive:
        os.makedirs(drive_dir, exist_ok=True)
        logger.info(f"Drive mounted: checkpoints will be mirrored to {drive_dir}")

    drive_log_path = os.path.join(drive_dir, 'training_log.csv') if use_drive else None

    csv_header = 'epoch,train_loss,val_loss,val_l1_loss,val_ssim_loss,val_psnr,val_ssim,val_wasserstein,val_rmse,val_kl_divergence,loss_change,psnr_realistic,learning_pattern\n'
    with open(log_path, 'w') as f:
        f.write(csv_header)
    if drive_log_path:
        with open(drive_log_path, 'w') as f:
            f.write(csv_header)

    # Track training behavior
    prev_train_loss = None
    final_metrics = {}

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        prog = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")

        for img, mask in prog:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(img, mask)
                loss, _ = criterion(output, img, mask)

            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss ({loss.item()}) at epoch {epoch} — skipping batch")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            prog.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        val_l1_loss = 0.0
        val_ssim_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        val_wasserstein = 0.0
        val_rmse = 0.0
        val_kl_divergence = 0.0
        val_num_samples = 0
        with torch.no_grad():
            for img, mask in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                img, mask = img.to(device), mask.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    output = model(img, mask)

                # Calculate validation loss
                loss, loss_components = criterion(output, img, mask)
                val_loss += loss.item()
                val_l1_loss += loss_components['l1_loss']
                val_ssim_loss += loss_components['ssim_loss']

                output = torch.clip(output, -1.0, 1.0)
                out_np = (output[:, 0].cpu().numpy() * 0.5 + 0.5) * 255.0  # (B, H, W)
                gt_np  = (img[:, 0].cpu().numpy()    * 0.5 + 0.5) * 255.0

                # Vectorised over batch — one numpy call instead of B calls
                mse_per  = ((out_np - gt_np) ** 2).mean(axis=(1, 2))         # (B,)
                val_psnr += np.where(mse_per == 0, 100.0,
                                     20 * np.log10(255.0 / np.sqrt(np.maximum(mse_per, 1e-10)))).sum()
                val_rmse += np.sqrt(mse_per).sum()

                # Per-sample metrics that don't support batch computation
                for o, g in zip(out_np, gt_np):
                    val_ssim += ssim_fn(o, g, data_range=255.0, channel_axis=None)
                    val_wasserstein += wasserstein_distance_2d(o, g)
                    val_kl_divergence += calculate_kl_divergence(o, g)
                    val_num_samples += 1

        val_loss /= len(val_loader)
        val_l1_loss /= len(val_loader)
        val_ssim_loss /= len(val_loader)
        if val_num_samples > 0:
            val_psnr /= val_num_samples
            val_ssim /= val_num_samples
            val_wasserstein /= val_num_samples
            val_rmse /= val_num_samples
            val_kl_divergence /= val_num_samples

        scheduler.step()

        logger.info(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_psnr={val_psnr:.2f} dB | val_ssim={val_ssim:.4f} | val_wasserstein={val_wasserstein:.2f} | val_rmse={val_rmse:.2f} | val_kl={val_kl_divergence:.3f}")

        # Store current epoch metrics
        current_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_l1_loss': val_l1_loss,
            'val_ssim_loss': val_ssim_loss,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
            'val_wasserstein': val_wasserstein,
            'val_rmse': val_rmse,
            'val_kl_divergence': val_kl_divergence
        }

        final_metrics = current_metrics  # Keep updating final_metrics

        # Call epoch callback for real-time monitoring (notebook integration)
        if epoch_callback:
            try:
                epoch_callback(epoch, current_metrics)
            except Exception as e:
                logger.warning(f"epoch_callback failed: {e}")

        loss_change = 0.0 if prev_train_loss is None else prev_train_loss - train_loss
        psnr_realistic = "realistic" if 30 <= val_psnr <= 45 else ("too_high" if val_psnr > 45 else "too_low")
        if epoch == 1:
            learning_pattern = "initial"
        elif loss_change > 0.01:
            learning_pattern = "good_learning"
        elif loss_change < 0.001:
            learning_pattern = "slow_learning"
        else:
            learning_pattern = "normal"

        csv_row = (f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_l1_loss:.4f},{val_ssim_loss:.4f},"
                   f"{val_psnr:.2f},{val_ssim:.4f},{val_wasserstein:.4f},{val_rmse:.4f},"
                   f"{val_kl_divergence:.4f},{loss_change:.4f},{psnr_realistic},{learning_pattern}\n")
        with open(log_path, 'a') as f:
            f.write(csv_row)
        if drive_log_path:
            with open(drive_log_path, 'a') as f:
                f.write(csv_row)

        prev_train_loss = train_loss

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_path = os.path.join(output_dir, 'best.pth')
            save_checkpoint(model, optimizer, epoch, best_path, metrics={'train_loss': train_loss})
            if use_drive:
                drive_best = os.path.join(drive_dir, 'best.pth')
                save_checkpoint(model, optimizer, epoch, drive_best, metrics={'train_loss': train_loss})

        if epoch % save_every == 0:
            epoch_path = os.path.join(output_dir, f'epoch_{epoch:03d}.pth')
            save_checkpoint(model, optimizer, epoch, epoch_path, metrics={'train_loss': train_loss})
            if use_drive:
                drive_epoch = os.path.join(drive_dir, f'epoch_{epoch:03d}.pth')
                save_checkpoint(model, optimizer, epoch, drive_epoch, metrics={'train_loss': train_loss})

            # Rotate old checkpoints
            if keep_checkpoints > 0:
                rotate_checkpoints(output_dir, keep_checkpoints)
                if use_drive:
                    rotate_checkpoints(drive_dir, keep_checkpoints)

    logger.info(f"Training complete. Best val PSNR: {best_val_psnr:.2f} dB")
    logger.info(f"Checkpoints in: {output_dir}/")

    return {
        'best_val_psnr': best_val_psnr,
        'log_path': log_path,
        'final_metrics': final_metrics,
        'output_dir': output_dir
    }
