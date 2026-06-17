"""
FAD-Net Inference -- OCTA Speckle Denoising
=============================================
Loads one or more noisy OCTA images, runs DDIM reverse sampling,
saves the denoised result(s).

Supports:
  - Single image:      --input / --output   (backward compatible)
  - Batch directory:   --input_dir / --output_dir [--gt_dir] [--save_metrics]
  - Batched inference: --batch_size N processes multiple images at once

The denoising pipeline (two-stage, Algorithm 1):
  1. Extract priors (F_h, sfe_proj) from I_noisy ONCE per input
  2. DDIM reverse sampling starting from x_T ~ N(0,I), reusing priors
"""

import os
import sys
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FADNet
from diffusion import ForwardDiffusion, ReverseDiffusion


# ============================================================================
# Metrics -- try evaluation.metrics, fall back to local implementations
# ============================================================================
_HAS_EVAL_METRICS = False
_eval_psnr_fn = None
_eval_ssim_fn = None

try:
    from evaluation.metrics import compute_psnr as _eval_psnr_fn
    from evaluation.metrics import compute_ssim as _eval_ssim_fn
    _HAS_EVAL_METRICS = True
except ImportError:
    try:
        import evaluation.metrics as _em
        _eval_psnr_fn = getattr(_em, 'compute_psnr', None)
        _eval_ssim_fn = getattr(_em, 'compute_ssim', None)
        if _eval_psnr_fn is not None and _eval_ssim_fn is not None:
            _HAS_EVAL_METRICS = True
    except ImportError:
        pass


def _compute_psnr_local(pred, target, data_range=1.0):
    """Simple PSNR: 20*log10(DR) - 10*log10(MSE)."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse.item() == 0:
        return float('inf')
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse.item()))


def _compute_ssim_local(pred, target, data_range=1.0,
                        window_size=11, sigma=1.5):
    """
    SSIM for single-channel (grayscale) images [B, 1, H, W].
    Computes per-image SSIM and returns the batch mean.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    B, C, H, W = pred.shape
    if C != 1:
        raise ValueError(
            f"SSIM local supports single-channel only; got C={C}. "
            f"Install evaluation.metrics for multi-channel support."
        )

    L = data_range
    K1, K2 = 0.01, 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # 2D Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords -= (window_size - 1) / 2.0
    g = torch.exp(-coords ** 2 / (2.0 * sigma ** 2))
    g = g / g.sum()
    window = torch.outer(g, g).unsqueeze(0).unsqueeze(0)  # [1, 1, w, w]

    ssim_vals = []
    for b in range(B):
        p = pred[b:b + 1]   # [1, 1, H, W]
        t = target[b:b + 1]

        mu1 = F.conv2d(p, window, padding=window_size // 2)
        mu2 = F.conv2d(t, window, padding=window_size // 2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(p * p, window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(t * t, window, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(p * t, window, padding=window_size // 2) - mu1_mu2

        num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = num / den
        ssim_vals.append(ssim_map.mean().item())

    return float(np.mean(ssim_vals))


def compute_psnr(pred, target, data_range=1.0):
    """Compute PSNR, delegating to evaluation.metrics if available."""
    if _HAS_EVAL_METRICS:
        return _eval_psnr_fn(pred, target, data_range=data_range)
    return _compute_psnr_local(pred, target, data_range)


def compute_ssim(pred, target, data_range=1.0):
    """Compute SSIM, delegating to evaluation.metrics if available."""
    if _HAS_EVAL_METRICS:
        return _eval_ssim_fn(pred, target, data_range=data_range)
    return _compute_ssim_local(pred, target, data_range)


# ============================================================================
# Image I/O helpers
# ============================================================================
_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _list_image_files(directory):
    """Return sorted list of absolute paths to image files under *directory*."""
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    paths = []
    for ext in _IMAGE_EXTS:
        paths.extend(root.rglob(f'*{ext}'))
        paths.extend(root.rglob(f'*{ext.upper()}'))
    return sorted(set(str(p) for p in paths))


def _load_image(path, image_size=512):
    """
    Load a single image as a CPU tensor in [-1, 1].

    Returns:
        tensor:    [1, 1, image_size, image_size]  float32
        orig_size: (W, H) of the original image
    """
    img = PILImage.open(path).convert('L')
    orig_size = img.size
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    tensor = transform(img).unsqueeze(0)  # [1, 1, H, W]
    return tensor, orig_size


def _tensor_to_image(tensor, orig_size=None):
    """
    Convert tensor in [-1, 1] to PIL Image uint8.

    Args:
        tensor:    [H, W], [1, H, W], or [1, 1, H, W]  in [-1, 1]
        orig_size: optional (W, H) to resize back to

    Returns:
        PIL Image in 'L' mode
    """
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        arr = tensor[0].cpu().numpy()
    elif tensor.ndim == 3:
        arr = tensor.squeeze(0).cpu().numpy()
    elif tensor.ndim == 2:
        arr = tensor.cpu().numpy()
    else:
        arr = tensor.squeeze().cpu().numpy()

    arr = (arr + 1.0) / 2.0
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    img = PILImage.fromarray(arr, mode='L')

    if orig_size is not None and img.size != orig_size:
        img = img.resize(orig_size, PILImage.BILINEAR)
    return img


# ============================================================================
# Core: denoise() -- refactored with optional pre-extracted priors
# ============================================================================
@torch.no_grad()
def denoise(model, noisy_img, diff_fwd, diff_rev,
            ddim_steps=50, device='cuda',
            F_h=None, sfe_proj=None, eta=0.0,
            t_start=None, rounds=1, median_ksize=0):
    """
    Denoise OCT image(s) using FAD-Net + DDIM sampling.

    Two modes:
      - Full reverse (t_start=None): start from x_T ~ N(0,I), full denoising.
      - Resampling  (t_start=int):   start from I_noisy + noise at t_start,
        then DDIM back to 0.  This preserves the original image structure
        and brightness while removing the added Gaussian noise component.
        Recommended for OCT: the model was trained to predict noise from
        x_t = sqrt(ᾱ_t)*I_noisy + sqrt(1-ᾱ_t)*ε, not from pure noise.
        Resampling at t_start ~ 400-600 gives the model an input close to
        its training distribution.

    Two stages:
      Stage 1 -- Extract priors (F_h, sfe_proj) from I_noisy ONCE.
      Stage 2 -- DDIM reverse sampling, guided by priors.

    Args:
        model:      FADNet instance
        noisy_img:  [B, 1, H, W] in [-1, 1]
        diff_fwd:   ForwardDiffusion instance
        diff_rev:   ReverseDiffusion instance (unused)
        ddim_steps: DDIM steps (default 50)
        device:     'cuda' or 'cpu'
        F_h, sfe_proj: pre-extracted priors (optional)
        eta:        DDIM stochasticity (0=deterministic, 0.1=small noise)
        t_start:    resampling timestep (None=start from pure noise;
                    e.g. 400-600 for OCT resampling denoising)

    Returns:
        denoised:  [B, 1, H, W] in [-1, 1]
    """
    model.eval()
    shape = noisy_img.shape
    B = shape[0]
    T_total = diff_fwd.T

    # ---- Stage 1: Extract priors ----
    if F_h is None or sfe_proj is None:
        F_h, sfe_proj = model.extract_priors(noisy_img)

    # Sanity check
    fh_batch = next(iter(F_h.values())).shape[0]
    if fh_batch != B:
        raise ValueError(
            f"Prior batch size {fh_batch} != noisy_img batch size {B}.")

    # ---- Stage 2: Initialize x_t ----
    if t_start is not None:
        # Resampling mode: start from I_noisy + Gaussian noise at t_start.
        #   x_start = sqrt(alpha_bar_start) * I_noisy + sqrt(1-alpha_bar_start) * eps
        # This matches the training distribution (Algorithm 1 line 6).
        t0 = max(1, min(t_start, T_total - 1))
        sqrt_ac_start = diff_fwd.sqrt_alphas_cumprod[t0]
        sqrt_omac_start = diff_fwd.sqrt_one_minus_alphas_cumprod[t0]
        eps_start = torch.randn(shape, device=device)
        x_t = sqrt_ac_start * noisy_img + sqrt_omac_start * eps_start
    else:
        # Full reverse: start from pure Gaussian noise x_T ~ N(0, I)
        t0 = T_total - 1
        x_t = torch.randn(shape, device=device)

    # DDIM timestep subsequence (uniform from t0 down to 0)
    step_indices = torch.linspace(t0, 0, ddim_steps,
                                   dtype=torch.long, device=device)

    for i in range(ddim_steps):
        t_curr = step_indices[i].item()
        t_next = step_indices[i + 1].item() if i < ddim_steps - 1 else -1
        t_batch = torch.full((B,), t_curr, device=device, dtype=torch.long)

        # Predict noise: eps_theta(x_t, t, F_h, sfe_proj)
        eps_pred = model.predict_noise(x_t, t_batch, F_h, sfe_proj)

        # Predicted clean image x_hat_0 (Eq.13)
        ac = diff_fwd.alphas_cumprod[t_curr]
        s_ac = diff_fwd.sqrt_alphas_cumprod[t_curr]
        s_omac = diff_fwd.sqrt_one_minus_alphas_cumprod[t_curr]

        x0_pred = (x_t - s_omac * eps_pred) / s_ac
        # NOTE: Do NOT clamp x0_pred during intermediate DDIM steps.
        # Clamping every step accumulates bias and causes the "foggy" /
        # low-contrast output common in diffusion-based denoising.
        # The model was trained without clamping; intermediate x0_pred
        # estimates may legitimately lie outside [-1,1] at high t.
        # Only clamp at the final output step.

        if t_next < 0:
            x_t = x0_pred.clamp(-1.0, 1.0)
            break

        # DDIM step with configurable stochasticity (eta)
        #   eta = 0  : deterministic DDIM (default, paper Fig.12)
        #   eta > 0  : stochastic DDIM (adds noise, improves sharpness)
        ac_next = diff_fwd.alphas_cumprod[t_next]
        sigma = eta * torch.sqrt((1.0 - ac_next) / (1.0 - ac)
                                 * (1.0 - ac / ac_next))
        dir_xt = torch.sqrt(1.0 - ac_next - sigma ** 2) * eps_pred
        x_t = torch.sqrt(ac_next) * x0_pred + dir_xt
        if eta > 0:
            x_t = x_t + sigma * torch.randn_like(x_t)

        if i % 10 == 0:
            print(f"  DDIM {i:3d}/{ddim_steps}  t={t_curr:4d}  "
                  f"x mean={x_t.mean().item():.4f}")

    return x_t


def _iterative_denoise(model, noisy_img, diff_fwd, diff_rev,
                       ddim_steps, device, eta, t_start, rounds):
    """
    Iterative resampling denoising.

    Each round: start from I_noisy + noise at t_start, DDIM denoise.
    The output becomes the new I_noisy for the next round.
    Priors are re-extracted each round from the current (partially denoised)
    image, providing increasingly clean structural guidance.
    """
    current = noisy_img
    for r in range(rounds):
        if rounds > 1:
            print(f"  Round {r+1}/{rounds}")
        current = denoise(model, current, diff_fwd, diff_rev,
                          ddim_steps=ddim_steps, device=device,
                          eta=eta, t_start=t_start,
                          rounds=1, median_ksize=0)
    return current


def _median_filter(tensor, kernel_size=3):
    """
    Apply median filter to a batch of images.

    Args:
        tensor:      [B, 1, H, W]
        kernel_size: int, must be odd

    Returns:
        filtered: [B, 1, H, W]
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    pad = kernel_size // 2
    B, C, H, W = tensor.shape
    # Unfold into patches and take median
    patches = F.unfold(tensor, kernel_size=kernel_size, padding=pad)
    patches = patches.view(B, C, kernel_size * kernel_size, H, W)
    median = patches.median(dim=2).values
    return median


# ============================================================================
# Batch Evaluation  --  evaluate()
# ============================================================================
def evaluate(model, input_dir, output_dir,
             diff_fwd, diff_rev,
             gt_dir=None,
             ddim_steps=50,
             image_size=512,
             batch_size=1,
             device='cuda',
             save_metrics=False,
             metrics_path=None):
    """
    Evaluate FAD-Net on all images in a directory.

    For each image:
      1. Load & preprocess to [-1, 1]
      2. Extract priors (once per batch)
      3. Run DDIM denoising
      4. Save denoised image to *output_dir*
      5. If *gt_dir* provided, compute PSNR / SSIM against matching ground truth

    Args:
        model:         FADNet instance
        input_dir:     Directory of noisy OCT images
        output_dir:    Directory to save denoised images
        diff_fwd:      ForwardDiffusion instance
        diff_rev:      ReverseDiffusion instance
        gt_dir:        Optional ground-truth directory (filenames must match)
        ddim_steps:    DDIM steps (default 50)
        image_size:    Resize dimension (default 512)
        batch_size:    Images to process at once (default 1)
        device:        'cuda' or 'cpu'
        save_metrics:  If True, save per-image metrics as JSON
        metrics_path:  Path for metrics JSON (default: output_dir/metrics.json)

    Returns:
        metrics dict:  {filename: {'psnr': float, 'ssim': float}, ...}
                       plus 'average' key with mean values when GT is available.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Gather input images ----
    input_paths = _list_image_files(input_dir)
    if not input_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")
    print(f"Found {len(input_paths)} image(s) in {input_dir}")

    # ---- Gather ground truth (if provided) ----
    gt_map = {}
    if gt_dir is not None:
        gt_paths = _list_image_files(gt_dir)
        for gp in gt_paths:
            gt_map[Path(gp).name] = gp
        if gt_map:
            print(f"Found {len(gt_map)} ground-truth image(s) in {gt_dir}")
        else:
            print(f"WARNING: No ground-truth images found in {gt_dir}")

    # ---- Transform pipeline (resize -> [0,1] -> [-1,1]) ----
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    # ---- Process in batches ----
    all_metrics = {}
    num_images = len(input_paths)
    num_batches = int(math.ceil(num_images / batch_size))

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_images)
        batch_paths = input_paths[start:end]

        # Load batch onto CPU first
        batch_tensors = []
        batch_orig_sizes = []
        for path in batch_paths:
            img = PILImage.open(path).convert('L')
            batch_orig_sizes.append(img.size)
            batch_tensors.append(transform(img))   # [1, H, W]

        noisy_batch = torch.stack(batch_tensors, dim=0).to(device)  # [B, 1, H, W]
        B_actual = noisy_batch.shape[0]

        print(f"\nBatch {batch_idx + 1}/{num_batches}: {B_actual} image(s)")

        # ---- Denoise the batch ----
        denoised_batch = denoise(
            model, noisy_batch, diff_fwd, diff_rev,
            ddim_steps=ddim_steps, device=device,
        )

        # ---- Save each image + compute metrics ----
        for b in range(B_actual):
            path = batch_paths[b]
            filename = Path(path).name
            stem = Path(path).stem

            # Convert to PIL and save
            denoised_tensor = denoised_batch[b]          # [1, 512, 512]
            denoised_img = _tensor_to_image(denoised_tensor,
                                            orig_size=batch_orig_sizes[b])
            out_path = os.path.join(output_dir, f"{stem}_denoised.png")
            denoised_img.save(out_path)

            # Metrics if ground truth is available for this filename
            metrics_entry = {}
            if filename in gt_map:
                gt_path = gt_map[filename]
                gt_pil = PILImage.open(gt_path).convert('L')
                gt_tensor = transform(gt_pil).unsqueeze(0).to(device)  # [1, 1, H, W]

                # Both are in [-1, 1]; convert to [0, 1] for metric computation
                pred_01 = (denoised_batch[b].unsqueeze(0) + 1.0) / 2.0
                gt_01 = (gt_tensor + 1.0) / 2.0

                psnr_val = compute_psnr(pred_01, gt_01, data_range=1.0)
                ssim_val = compute_ssim(pred_01, gt_01, data_range=1.0)
                metrics_entry = {
                    'psnr': round(float(psnr_val), 4),
                    'ssim': round(float(ssim_val), 6),
                }
                print(f"  {filename}: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}")
            else:
                print(f"  {filename}: saved (no GT)")

            all_metrics[filename] = metrics_entry

    # ---- Compute averages ----
    psnr_vals = [v['psnr'] for v in all_metrics.values() if 'psnr' in v]
    ssim_vals = [v['ssim'] for v in all_metrics.values() if 'ssim' in v]

    if psnr_vals:
        all_metrics['average'] = {
            'psnr': round(float(np.mean(psnr_vals)), 4),
            'ssim': round(float(np.mean(ssim_vals)), 6),
        }
        print(f"\nAverage over {len(psnr_vals)} image(s): "
              f"PSNR={all_metrics['average']['psnr']:.2f} dB, "
              f"SSIM={all_metrics['average']['ssim']:.4f}")

    # ---- Save metrics JSON ----
    if save_metrics or metrics_path:
        if metrics_path is None:
            metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")

    return all_metrics


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='FAD-Net OCTA Denoising -- Single Image + Batch Evaluation'
    )

    # Single-image mode (backward compatible)
    parser.add_argument('--input', type=str, default=None,
                        help='Path to a single noisy OCTA image')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for a single denoised image')

    # Batch evaluation mode
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory of noisy images to denoise')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save denoised images')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Optional ground truth directory (filenames must match)')
    parser.add_argument('--save_metrics', action='store_true', default=False,
                        help='Save per-image metrics to JSON (output_dir/metrics.json)')
    parser.add_argument('--metrics_path', type=str, default=None,
                        help='Custom path for the metrics JSON file')

    # Common options
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM stochasticity (default 0=deterministic)')
    parser.add_argument('--t_start', type=int, default=500,
                        help='Resampling timestep (400-600 recommended)')
    parser.add_argument('--rounds', type=int, default=1,
                        help='Iterative denoising rounds. Each round uses the '
                             'previous output as new I_noisy. 2-3 rounds '
                             'progressively removes more speckle.')
    parser.add_argument('--median', type=int, default=0,
                        help='Median filter kernel size (odd, e.g. 3). '
                             '0 = disabled.')
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='DDIM steps (paper Fig.12: 30-50)')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image resize dimension (default 512)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Images to process at once in batch mode (default 1)')

    args = parser.parse_args()

    # ---- Resolve mode ----
    single_mode = (args.input is not None or args.output is not None)
    batch_mode = (args.input_dir is not None or args.output_dir is not None)

    if single_mode and batch_mode:
        print("ERROR: Cannot use both single-image (--input/--output) "
              "and batch (--input_dir/--output_dir) modes simultaneously.")
        sys.exit(1)

    if not single_mode and not batch_mode:
        parser.print_help()
        sys.exit(1)

    if single_mode:
        if not args.input or not args.output:
            print("ERROR: --input and --output are both required "
                  "for single-image mode.")
            sys.exit(1)

    # ---- Device ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Build model ----
    print("Building FAD-Net...")
    model = FADNet(in_channels=1, sfe_frozen=True).to(device)

    if args.checkpoint:
        print(f"  Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get('ema_state_dict', ckpt['model_state_dict'])
        model.load_state_dict(state_dict, strict=False)
        used = 'EMA' if 'ema_state_dict' in ckpt else 'standard'
        print(f"  Loaded from epoch {ckpt.get('epoch', '?')} ({used} weights)")
    else:
        print("  WARNING: No checkpoint -- using RANDOM weights!")
        print("  Output will be random noise, not meaningful denoising.")
        print("  Train the model first, then pass --checkpoint path.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ---- Build diffusion ----
    diff_fwd = ForwardDiffusion(T=1000).to(device)
    diff_rev = ReverseDiffusion(diff_fwd).to(device)

    # ================================================================
    # Single-Image Mode (backward compatible)
    # ================================================================
    if single_mode:
        print(f"\nLoading: {args.input}")
        noisy_tensor, orig_size = _load_image(args.input, args.image_size)
        noisy_tensor = noisy_tensor.to(device)

        print(f"  Tensor: {list(noisy_tensor.shape)}  "
              f"range=[{noisy_tensor.min().item():.4f}, "
              f"{noisy_tensor.max().item():.4f}]")

        print(f"\nExtracting priors from I_noisy (DWT+MHFB+SFE -- once)...")
        print(f"Running DDIM denoising ({args.ddim_steps} steps, eta={args.eta})...")
        t_start = args.t_start if args.t_start > 0 else None

        # Iterative resampling denoising
        if args.rounds > 1:
            denoised = _iterative_denoise(model, noisy_tensor, diff_fwd, diff_rev,
                                          ddim_steps=args.ddim_steps, device=device,
                                          eta=args.eta, t_start=t_start,
                                          rounds=args.rounds)
        else:
            denoised = denoise(model, noisy_tensor, diff_fwd, diff_rev,
                               ddim_steps=args.ddim_steps, device=device,
                               eta=args.eta, t_start=t_start)

        # Optional median filter post-processing
        if args.median > 0:
            denoised = _median_filter(denoised, kernel_size=args.median)
            print(f"  Median filter: kernel={args.median}")

        print(f"\nDenoised: {list(denoised.shape)}  "
              f"range=[{denoised.min().item():.4f}, "
              f"{denoised.max().item():.4f}]")

        # Save
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        denoised_img = _tensor_to_image(denoised[0], orig_size=orig_size)
        denoised_img.save(args.output)
        print(f"Saved: {args.output} "
              f"({denoised_img.size[0]}x{denoised_img.size[1]})")

    # ================================================================
    # Batch Evaluation Mode
    # ================================================================
    else:
        _output_dir = args.output_dir
        if _output_dir is None:
            _output_dir = str(Path(args.input_dir).parent /
                              (Path(args.input_dir).name + '_denoised'))

        evaluate(
            model=model,
            input_dir=args.input_dir,
            output_dir=_output_dir,
            diff_fwd=diff_fwd,
            diff_rev=diff_rev,
            gt_dir=args.gt_dir,
            ddim_steps=args.ddim_steps,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=device,
            save_metrics=args.save_metrics,
            metrics_path=args.metrics_path,
        )


if __name__ == "__main__":
    main()
