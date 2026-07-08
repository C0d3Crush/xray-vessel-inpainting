# -*- coding: utf-8 -*-
import argparse

from trainer import train_model  # re-exported for backward compatibility (e.g. notebook imports)


def main():
    parser = argparse.ArgumentParser(description="Train CMT on ARCADE grayscale X-rays")
    parser.add_argument('--train_img',        default='data/arcade/syntax/train/images',      help='Path to training images directory')
    parser.add_argument('--train_ann',        default='data/arcade/syntax/train/annotations/train.json', help='Path to training COCO annotation JSON')
    parser.add_argument('--train_mask',       default=None,                                   help='Path to precomputed training masks directory (skips on-the-fly generation)')
    parser.add_argument('--val_img',          default='data/arcade/syntax/val/images',        help='Path to validation images directory')
    parser.add_argument('--val_ann',          default='data/arcade/syntax/val/annotations/val.json', help='Path to validation COCO annotation JSON')
    parser.add_argument('--val_mask',         default=None,                                   help='Path to precomputed validation masks directory')
    parser.add_argument('--output_dir',       default='checkpoints',                          help='Directory to save checkpoints and training log')
    parser.add_argument('--ckpt',             default=None,                                   help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs',           type=int,   default=100,                        help='Number of training epochs')
    parser.add_argument('--batch_size',       type=int,   default=4,                          help='Batch size for training and validation')
    parser.add_argument('--lr',               type=float, default=1e-4,                       help='Initial learning rate for Adam optimizer')
    parser.add_argument('--num_workers',      type=int,   default=2,                          help='Number of DataLoader worker processes')
    parser.add_argument('--save_every',       type=int,   default=10,                         help='Save an epoch checkpoint every N epochs')
    parser.add_argument('--keep_checkpoints', type=int,   default=3,                          help='Number of epoch checkpoints to keep (oldest deleted)')
    parser.add_argument('--device',           default='cpu', choices=['cpu', 'cuda'],          help='Device to train on')
    parser.add_argument('--smoke_test',       action='store_true',                            help='Run a minimal smoke test (1 epoch, tiny dataset) to verify the pipeline')
    parser.add_argument('--smoke_size',       type=int,   default=2,                          help='Number of samples to use during smoke test')
    parser.add_argument('--input_size',       type=int,   default=256,                        help='Patch size in pixels (must be power of 2, min 64)')
    parser.add_argument('--random_masks',     action='store_true',                            help='Use random vessel-padded masks instead of COCO annotations')
    parser.add_argument('--mask_padding',     type=int,   default=10,                         help='Dilation radius in pixels applied to vessel masks')
    parser.add_argument('--ssim_weight',      type=float, default=0.5,                        help='Weight of SSIM loss component')
    parser.add_argument('--mask_weight',      type=float, default=6.0,                        help='L1 loss weight on masked (vessel) regions')
    parser.add_argument('--perceptual_weight',type=float, default=0.1,                        help='Weight for VGG16 perceptual loss (0 = disabled)')
    parser.add_argument('--adv_weight',       type=float, default=0.0,                        help='Weight for adversarial hinge loss with PatchGAN discriminator (0 = disabled)')
    parser.add_argument('--gan_start_epoch',  type=int,   default=1,                          help='Epoch from which the adversarial loss becomes active (warm-up)')
    parser.add_argument('--patches_per_image',type=int,   default=4,                          help='Number of patches extracted per image per epoch')
    parser.add_argument('--foreground_prob',  type=float, default=0.75,                       help='Probability of sampling patches centered on foreground (vessel/background) regions')
    parser.add_argument('--max_shapes',       type=int,   default=5,                          help='Maximum number of random shapes added to generated masks')
    parser.add_argument('--vessel_safe_training', action='store_true',                        help='Generate vessel-safe background masks (no overlap with vessel annotations)')
    parser.add_argument('--no_background_training', dest='background_training', action='store_false', help='Disable background training mode (use vessel masks instead)')
    parser.add_argument('--drive_dir',           default=None,                                help='Google Drive directory to mirror checkpoints into (Colab only)')
    parser.add_argument('--amp',                 action='store_true',                         help='Enable automatic mixed precision training (CUDA only, ~30-40%% speedup)')
    parser.set_defaults(background_training=True)
    args = parser.parse_args()

    train_model(
        train_img=args.train_img, train_ann=args.train_ann, train_mask=args.train_mask,
        val_img=args.val_img,     val_ann=args.val_ann,     val_mask=args.val_mask,
        output_dir=args.output_dir, ckpt=args.ckpt,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        num_workers=args.num_workers, save_every=args.save_every,
        keep_checkpoints=args.keep_checkpoints, device=args.device,
        smoke_test=args.smoke_test, smoke_size=args.smoke_size,
        input_size=args.input_size, random_masks=args.random_masks,
        mask_padding=args.mask_padding, ssim_weight=args.ssim_weight,
        mask_weight=args.mask_weight,
        perceptual_weight=args.perceptual_weight,
        adv_weight=args.adv_weight, gan_start_epoch=args.gan_start_epoch,
        patches_per_image=args.patches_per_image, foreground_prob=args.foreground_prob,
        max_shapes=args.max_shapes, vessel_safe_training=args.vessel_safe_training,
        background_training=args.background_training, drive_dir=args.drive_dir,
        amp=args.amp,
    )


if __name__ == '__main__':
    main()
