"""
FAD-Net Training Pipeline
==========================
Paper: FAD-Net, IEEE TIM 2026, Algorithm 1, Section III-D

Pipeline:
    1. DWT noisy OCT image -> LL, LH, HL, HH
    2. MHFB processes {LH, HL, HH} -> F_h (4-scale HF features)
    3. SFE extracts structural features from original image
    4. Forward diffusion: x_t = sqrt(alpha_bar_t) * I_noisy + sqrt(...) * noise
    5. Timestep embedding -> TAWG -> W_t
    6. U-Net encoder -> skip features; CrossAttn(SFE) -> F_enc
    7. F_fuse = (1-W_t) * F_enc + W_t * F_h  (per scale)
    8. U-Net decoder with F_fuse injection -> eps_pred
    9. Loss = ||eps_pred - noise||^2  (Eq.14)

Usage:
    python train.py --data_dir /path/to/oct/images --epochs 100
"""

import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image as PILImage
from tqdm import tqdm

# Import FAD-Net modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dwt_module import HaarDWT, get_high_frequency_subbands
from mhfb import MHFB
from sfe import SFE, SFEProjection
from tawg import TAWG, TimestepEmbedding
from cross_attention import CrossAttention
from unet import UNet
from diffusion import ForwardDiffusion
from model import FADNet


# ============================================================
# Dataset
# ============================================================
class OCTDataset(Dataset):
    """
    Simple dataset for OCT/OCTA images.

    Expects a directory of image files (.png, .jpg, .tif, .bmp).
    Images are loaded as grayscale, resized to target_size, normalized to [0, 1].

    Args:
        data_dir:     path to directory containing OCT images
        target_size:  (H, W) target resolution (default 512x512, paper Sec IV-A2)
    """

    def __init__(self, data_dir, target_size: int = 512,
                 crop_size: int = 0, data_repeat: int = 1):
        """
        Args:
            data_dir:    str or list of str — path(s) to OCT image directories.
                         Multiple dirs are pooled into one dataset.
            target_size: resize to this before crop (default 512)
            crop_size:   if > 0, random-crop to crop_size×crop_size.
        """
        self.target_size = target_size
        self.crop_size = crop_size
        self.data_repeat = data_repeat
        self._n_original = 0  # set after loading paths
        # Normalize to list
        if isinstance(data_dir, str):
            data_dir = [data_dir]
        self.data_dirs = [Path(d) for d in data_dir]

        exts = {'*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp'}
        self.paths = []
        for d in self.data_dirs:
            for ext in exts:
                self.paths.extend(sorted(d.rglob(ext)))
        self.paths = [str(p) for p in self.paths]

        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found in {data_dir}")

        self._n_original = len(self.paths)

        self.base_transform = transforms.Compose([
            transforms.Resize((target_size, target_size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),                                 # -> [0, 1]
            transforms.Normalize(mean=[0.5], std=[0.5]),           # -> [-1, 1]
        ])

        # Augmentations for unsupervised training (no labels needed)
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-10, 10),
                                       interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        # Pixel-level dropout: randomly erase individual pixels and let
        # the model predict them from context.  This is a self-supervised
        # inpainting task — exactly the same skill needed for speckle
        # denoising (inferring clean pixel values from noisy neighbors).
        self.pixel_dropout_prob = 0.03     # random noise pixel dropout
        self.neighbor_erase_prob = 0.10    # neighbourhood prediction erasure
                                            # Reduced from 0.3→0.1 to preserve
                                            # edge sharpness (N2V over-smoothing)

    def __len__(self):
        return len(self.paths) * self.data_repeat

    def __getitem__(self, idx):
        # Each repeat cycle uses same paths but different random augmentations
        idx = idx % self._n_original
        try:
            img = PILImage.open(self.paths[idx]).convert('L')
        except Exception:
            # Corrupted / unreadable file — fall back to a different image
            fallback = (idx + 1) % self._n_original
            img = PILImage.open(self.paths[fallback]).convert('L')
        tensor = self.base_transform(img)  # [1, H, W] in [-1, 1]

        # Data augmentation: geometric + pixel dropout
        tensor = self.augment(tensor)          # flip, rotate
        # Pixel dropout: randomly erase individual pixels
        if self.pixel_dropout_prob > 0:
            mask = torch.rand_like(tensor) < self.pixel_dropout_prob
            noise = torch.randn_like(tensor) * 0.1  # small random fill
            tensor = torch.where(mask, noise, tensor)

        # Neighborhood Prediction Erasure: replace random pixels with
        # their 3x3 neighbourhood average.  The model must infer the
        # original value from surrounding context — directly training
        # the local inference skill needed for speckle removal.
        # Vascular continuity: if neighbours are vessels, the centre
        # should be a vessel; if neighbours are background, centre
        # should be background.  This prior is embedded in the
        # neighbourhood average fill.
        if self.neighbor_erase_prob > 0:
            blind_mask = torch.rand_like(tensor) < self.neighbor_erase_prob
            # 3x3 average pooling to get neighbourhood mean (N2V blind-spot)
            kernel = torch.ones(1, 1, 3, 3, device=tensor.device) / 9.0
            neighbor_avg = torch.nn.functional.conv2d(
                tensor, kernel, padding=1)
            tensor = torch.where(blind_mask, neighbor_avg, tensor)
        else:
            blind_mask = torch.zeros_like(tensor, dtype=torch.bool)

        tensor = tensor.clamp(-1.0, 1.0)       # keep in DDPM range

        # Random crop → more samples, smaller memory
        if self.crop_size > 0 and self.crop_size < self.target_size:
            _, H, W = tensor.shape
            top = torch.randint(0, H - self.crop_size + 1, (1,)).item()
            left = torch.randint(0, W - self.crop_size + 1, (1,)).item()
            tensor = tensor[:, top:top + self.crop_size,
                            left:left + self.crop_size]
            blind_mask = blind_mask[:, top:top + self.crop_size,
                                    left:left + self.crop_size]

        return tensor, blind_mask


# ============================================================
# Training Utilities
# ============================================================
def save_checkpoint(model, optimizer, scaler, epoch, loss, path,
                    ema_model=None):
    """Save training checkpoint (optionally with EMA weights)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
    }
    if ema_model is not None:
        ckpt['ema_state_dict'] = ema_model.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(model, optimizer, scaler, path, device):
    """Load training checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scaler is not None and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('loss', float('inf'))


# ============================================================
# Validation
# ============================================================
def run_validation(model, diff_fwd, device, args, epoch, writer,
                   output_dir, logger, use_amp):
    """
    Validate the model on images from --val_dir using abbreviated DDIM sampling.

    - Loads up to 4 images from val_dir
    - Runs 20-step DDIM denoising (deterministic, eta=0)
    - Logs noisy/denoised images to TensorBoard
    - If --gt_dir is provided and filenames match, computes PSNR/SSIM
    - Saves denoised samples to output_dir/val_samples/epoch_NNN/

    Uses the two-stage inference pattern (Algorithm 1):
      Stage 1: extract_priors(I_noisy) once -> F_h, sfe_proj
      Stage 2: predict_noise(x_t, t, F_h, sfe_proj) per DDIM step
    """
    val_dir = Path(args.val_dir)
    exts = {'*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp'}
    val_paths = []
    for ext in exts:
        val_paths.extend(sorted(val_dir.rglob(ext)))
    val_paths = [str(p) for p in val_paths[:4]]  # max 4 for speed

    if len(val_paths) == 0:
        logger.warning(f"No images found in val_dir: {args.val_dir}")
        return

    # ---- Output directory ----
    val_sample_dir = output_dir / 'val_samples' / f'epoch_{epoch + 1:03d}'
    val_sample_dir.mkdir(parents=True, exist_ok=True)

    # ---- Ground-truth lookup by stem ----
    gt_lookup = {}
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
        for ext in exts:
            for gp in gt_dir.rglob(ext):
                gt_lookup[gp.stem] = str(gp)

    # ---- Image transform (same normalisation as training) ----
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # ---- Metric imports (three fallback levels) ----
    has_metrics = False
    compute_psnr_fn = None
    compute_ssim_fn = None

    # Level 1: project evaluation module
    try:
        from evaluation.metrics import compute_psnr, compute_ssim
        compute_psnr_fn = compute_psnr
        compute_ssim_fn = compute_ssim
        has_metrics = True
    except ImportError:
        pass

    # Level 2: scikit-image
    if not has_metrics:
        try:
            from skimage.metrics import peak_signal_noise_ratio as _psnr
            from skimage.metrics import structural_similarity as _ssim
            compute_psnr_fn = _psnr
            compute_ssim_fn = _ssim
            has_metrics = True
        except ImportError:
            pass

    # Level 3: torchmetrics
    if not has_metrics:
        try:
            from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
            _psnr_tm = PeakSignalNoiseRatio(data_range=1.0).to(device)
            _ssim_tm = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

            def compute_psnr_fn(gt, pred, data_range=1.0):
                gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)
                pr_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).to(device)
                return _psnr_tm(pr_t, gt_t).item()

            def compute_ssim_fn(gt, pred, data_range=1.0):
                gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)
                pr_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).to(device)
                return _ssim_tm(pr_t, gt_t).item()

            has_metrics = True
        except ImportError:
            pass

    # ---- DDIM configuration ----
    ddim_steps = 20
    T = args.T
    step_indices = torch.linspace(T - 1, 0, ddim_steps,
                                   dtype=torch.long, device=device)

    all_psnr = []
    all_ssim = []

    model.eval()
    with torch.no_grad():
        for idx, vp in enumerate(val_paths):
            vp_stem = Path(vp).stem
            logger.info(f"  Val [{idx + 1}/{len(val_paths)}]: {vp_stem}")

            # ---- Load noisy image ----
            img = PILImage.open(vp).convert('L')
            I_noisy = val_transform(img).unsqueeze(0).to(device)  # [1, 1, H, W]

            # ---- Stage 1: extract priors once (Algorithm 1, lines 2-3) ----
            F_h, sfe_proj = model.extract_priors(I_noisy)

            # ---- Stage 2: DDIM reverse sampling (Algorithm 1, lines 7-9) ----
            x_t = torch.randn_like(I_noisy)
            for i in range(ddim_steps):
                t_curr = step_indices[i].item()
                t_next = step_indices[i + 1].item() if i < ddim_steps - 1 else -1
                t_batch = torch.tensor([t_curr], device=device, dtype=torch.long)

                eps_pred = model.predict_noise(x_t, t_batch, F_h, sfe_proj)

                ac = diff_fwd.alphas_cumprod[t_curr]
                s_ac = diff_fwd.sqrt_alphas_cumprod[t_curr]
                s_omac = diff_fwd.sqrt_one_minus_alphas_cumprod[t_curr]

                # Predicted clean image (Eq.13)
                x0_pred = (x_t - s_omac * eps_pred) / s_ac
                x0_pred = x0_pred.clamp(-1.0, 1.0)

                if t_next < 0:
                    x_t = x0_pred
                    break

                # DDIM deterministic step (eta = 0)
                ac_next = diff_fwd.alphas_cumprod[t_next]
                dir_xt = torch.sqrt(1.0 - ac_next) * eps_pred
                x_t = torch.sqrt(ac_next) * x0_pred + dir_xt

            denoised_tensor = x_t  # [1, 1, H, W]

            # ---- Postprocess: [-1,1] -> [0,1] ----
            noisy_01 = (I_noisy + 1.0) / 2.0
            denoised_01 = (denoised_tensor + 1.0) / 2.0

            # ---- Save denoised image ----
            denoised_np = denoised_01.squeeze().cpu().numpy()
            denoised_np = np.clip(denoised_np * 255.0, 0, 255).astype(np.uint8)
            PILImage.fromarray(denoised_np, mode='L').save(
                str(val_sample_dir / f"{idx:02d}_{vp_stem}_denoised.png"))

            # ---- Save noisy reference ----
            noisy_np = noisy_01.squeeze().cpu().numpy()
            noisy_np = np.clip(noisy_np * 255.0, 0, 255).astype(np.uint8)
            PILImage.fromarray(noisy_np, mode='L').save(
                str(val_sample_dir / f"{idx:02d}_{vp_stem}_noisy.png"))

            # ---- TensorBoard ----
            writer.add_image(f'val/{idx:02d}_{vp_stem}/noisy', noisy_01, epoch)
            writer.add_image(f'val/{idx:02d}_{vp_stem}/denoised', denoised_01, epoch)

            # ---- Metrics against ground truth ----
            if vp_stem in gt_lookup:
                gt_path = gt_lookup[vp_stem]
                gt_img = PILImage.open(gt_path).convert('L')
                gt_tensor = val_transform(gt_img).unsqueeze(0).to(device)
                gt_01 = (gt_tensor + 1.0) / 2.0

                writer.add_image(f'val/{idx:02d}_{vp_stem}/gt', gt_01, epoch)

                if has_metrics and compute_psnr_fn is not None:
                    dn_np = denoised_01.squeeze().cpu().numpy()
                    gt_np = gt_01.squeeze().cpu().numpy()

                    # Ensure same spatial dimensions (paranoid safeguard)
                    if dn_np.shape != gt_np.shape:
                        gt_pil = PILImage.fromarray(
                            (gt_np * 255).astype(np.uint8))
                        gt_resized = gt_pil.resize(
                            (dn_np.shape[1], dn_np.shape[0]),
                            PILImage.BILINEAR)
                        gt_np = np.array(gt_resized).astype(np.float32) / 255.0

                    try:
                        psnr_val = compute_psnr_fn(gt_np, dn_np, data_range=1.0)
                        ssim_val = compute_ssim_fn(gt_np, dn_np, data_range=1.0)
                        all_psnr.append(psnr_val)
                        all_ssim.append(ssim_val)
                        logger.info(f"    PSNR={psnr_val:.4f}  SSIM={ssim_val:.4f}")
                    except Exception as e:
                        logger.warning(f"    Metric error on {vp_stem}: {e}")

    # ---- Epoch-level averages ----
    if all_psnr:
        avg_psnr = float(np.mean(all_psnr))
        writer.add_scalar('val/avg_psnr', avg_psnr, epoch)
        logger.info(f"  Avg PSNR: {avg_psnr:.4f}  ({len(all_psnr)} images)")
    if all_ssim:
        avg_ssim = float(np.mean(all_ssim))
        writer.add_scalar('val/avg_ssim', avg_ssim, epoch)
        logger.info(f"  Avg SSIM: {avg_ssim:.4f}  ({len(all_ssim)} images)")

    logger.info(f"  Samples saved to: {val_sample_dir}")


# ============================================================
# Main Training Script
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='FAD-Net Training')
    parser.add_argument('--data_dir', type=str, nargs='+', required=True,
                        help='Path(s) to OCT image directories. '
                             'Multiple dirs = pooled together. '
                             'Example: --data_dir D:/lyx/octa/OCTA(ILM_OPL) D:/lyx/octa/OCTA(OPL_BM)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (paper: 2, Section IV-A2)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (paper: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (paper: 1e-4, Section IV-A2)')
    parser.add_argument('--T', type=int, default=1000,
                        help='Diffusion timesteps (DDPM standard)')
    parser.add_argument('--image_size', type=int, default=300,
                        help='Input image resize (paper: 512, Section IV-A2)')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='Random crop size. 0=no crop. 256 recommended '
                             'for 600-image datasets: 4x samples, batch_size=4.')
    parser.add_argument('--data_repeat', type=int, default=1,
                        help='Repeat dataset N times per epoch (default 1). '
                             'Each image appears N times with different '
                             'augmentations each time.')
    parser.add_argument('--grad_accum', type=int, default=4,
                        help='Gradient accumulation steps. Reduces noise in '
                             'batch_size=1 training.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Log every N steps')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--sfe_checkpoint', type=str, default=None,
                        help='Path to pre-trained SFE weights (optional)')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Directory of validation images (skips validation if not set)')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Directory of ground truth images for PSNR/SSIM (optional)')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Run validation every N epochs (default: 5)')
    args = parser.parse_args()

    # ---- Setup ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")

    # ---- Dataset ----
    logger.info(f"Loading dataset from: {args.data_dir}")
    dataset = OCTDataset(args.data_dir, target_size=args.image_size,
                         crop_size=args.crop_size,
                         data_repeat=args.data_repeat)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers,
                            drop_last=True, pin_memory=True)
    logger.info(f"Dataset size: {len(dataset)} images")

    # ---- Model ----
    logger.info("Building FAD-Net...")
    model = FADNet(in_channels=1).to(device)

    # Load pre-trained SFE if provided
    if args.sfe_checkpoint:
        logger.info(f"Loading SFE weights from {args.sfe_checkpoint}")
        sfe_state = torch.load(args.sfe_checkpoint, map_location=device,
                               weights_only=False)
        if 'sfe_state_dict' in sfe_state:
            model.sfe.load_state_dict(sfe_state['sfe_state_dict'])
        else:
            model.sfe.load_state_dict(sfe_state, strict=False)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {n_params:,}")
    logger.info(f"Trainable parameters: {n_trainable:,}")

    # ---- Diffusion ----
    diff = ForwardDiffusion(T=args.T).to(device)

    # ---- Optimizer (paper Section IV-A2) ----
    #   "Adam optimizer with momentum parameters β1=0.5 and β2=0.999"
    #   Learning rate 1e-4.  No weight decay mentioned in the paper.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, betas=(0.5, 0.999),
    )

    # ---- Mixed Precision ----
    scaler = GradScaler('cuda')
    use_amp = (device.type == 'cuda')

    # ---- EMA (Exponential Moving Average) ----
    # Standard DDPM practice: maintain EMA of model weights for inference.
    # Improves sample quality especially with limited data.
    ema_decay = 0.9999
    ema_model = FADNet(in_channels=1).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False

    def update_ema():
        with torch.no_grad():
            for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(ema_decay).add_(model_p.data, alpha=1 - ema_decay)

    # ---- Resume ----
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        start_epoch, best_loss = load_checkpoint(
            model, optimizer, scaler, args.resume, device
        )
        # Restore EMA if available
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if 'ema_state_dict' in ckpt:
            ema_model.load_state_dict(ckpt['ema_state_dict'])
            logger.info("  Restored EMA weights")
        start_epoch += 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # ---- Training Loop ----
    global_step = start_epoch * len(dataloader)
    logger.info(f"Starting training: {args.epochs} epochs, "
                f"{len(dataloader)} batches/epoch")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, (images, blind_mask) in enumerate(pbar):
            images = images.to(device)               # [B, 1, H, W]
            blind_mask = blind_mask.to(device)        # [B, 1, H, W] bool

            # ---- Sample random timesteps ----
            t = torch.randint(0, args.T, (images.size(0),), device=device)

            # ---- Forward diffusion (Eq.10 / Algorithm 1 line 6) ----
            #   x_t = sqrt(alpha_bar_t) * I_noisy + sqrt(1-alpha_bar_t) * epsilon
            x_t, noise = diff(images, t)

            # ---- FAD-Net forward (Algorithm 1 lines 2-9) ----
            with autocast('cuda', enabled=use_amp):
                eps_pred = model(images, x_t, t)
                # Main loss: MSE on all pixels (Eq.14)
                loss_main = F.mse_loss(eps_pred, noise)
                # N2V blind-spot loss: extra weight on neighbor-erased pixels
                # Forces model to predict noise from context at masked locations
                if blind_mask.any():
                    loss_blind = F.mse_loss(
                        eps_pred[blind_mask], noise[blind_mask])
                    loss = loss_main + 0.5 * loss_blind  # λ=0.5
                else:
                    loss = loss_main

            # ---- Backward (gradient accumulation) ----
            loss = loss / args.grad_accum
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Step only after accumulating enough gradients
            if (batch_idx + 1) % args.grad_accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Update EMA after each optimizer step
            update_ema()

            epoch_loss += loss.item()
            global_step += 1

            # ---- Logging ----
            if batch_idx % args.log_interval == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr',
                                  optimizer.param_groups[0]['lr'], global_step)
                # Log TAWG weights for each scale
                t_sample = torch.tensor([100, 500, 900], device=device)
                t_emb_sample = model.time_embed(t_sample)
                with torch.no_grad():
                    W_sample = model.tawg(t_emb_sample)  # [3, 4]
                for si in range(4):
                    writer.add_scalar(f'tawg/scale{si+1}_W_t@t100', W_sample[0, si].item(), global_step)
                    writer.add_scalar(f'tawg/scale{si+1}_W_t@t500', W_sample[1, si].item(), global_step)
                    writer.add_scalar(f'tawg/scale{si+1}_W_t@t900', W_sample[2, si].item(), global_step)

            # ---- Progress bar ----
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{epoch_loss / (batch_idx + 1):.4f}',
            })

        # ---- Epoch summary ----
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        logger.info(f"Epoch {epoch+1}/{args.epochs} -- avg loss: {avg_loss:.6f}")

        # ---- Save every epoch (no risk of losing progress) ----
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        ckpt_path = ckpt_dir / f"fadnet_epoch{epoch+1:03d}.pt"
        save_checkpoint(model, optimizer, scaler, epoch, avg_loss,
                        str(ckpt_path), ema_model=ema_model)
        logger.info(f"  Saved: {ckpt_path} {'(best)' if is_best else ''}")

        if is_best:
            best_path = ckpt_dir / "best_model.pt"
            save_checkpoint(model, optimizer, scaler, epoch, avg_loss,
                            str(best_path), ema_model=ema_model)

        latest_path = ckpt_dir / "latest.pt"
        save_checkpoint(model, optimizer, scaler, epoch, avg_loss,
                        str(latest_path), ema_model=ema_model)

        # ---- Validation (if configured) ----
        if args.val_dir is not None and (epoch + 1) % args.val_interval == 0:
            logger.info(f"Running validation at epoch {epoch + 1}...")
            run_validation(model, diff, device, args, epoch, writer,
                           output_dir, logger, use_amp)
            model.train()  # ensure back in train mode after validation

    writer.close()
    logger.info(f"Training complete. Best loss: {best_loss:.6f}")
    logger.info(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
