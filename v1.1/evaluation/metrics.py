"""
Evaluation metrics for FAD-Net, as described in:
  "FAD-Net: Frequency-Aware Decomposition Network for Medical Image Denoising"
  IEEE TIM 2026, Section IV-B, Equations (15)-(20).

Metrics implemented:
  - PSNR  (Eq.15-16): Peak Signal-to-Noise Ratio
  - SSIM  (Eq.17):    Structural Similarity Index Measure
  - CNR   (Eq.18):    Contrast-to-Noise Ratio
  - ENL   (Eq.19):    Equivalent Number of Looks
  - EPI   (Eq.20):    Edge Preservation Index

All functions operate on PyTorch tensors of shape [B, 1, H, W] by default.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _check_inputs(*tensors: torch.Tensor) -> None:
    """Verify that all tensors are on the same device and have compatible shapes."""
    ref_shape = tensors[0].shape
    ref_device = tensors[0].device
    for i, t in enumerate(tensors):
        if t.dim() != 4:
            raise ValueError(
                f"Expected 4-D tensor [B, C, H, W], got {t.dim()}-D tensor: {t.shape}"
            )
        if t.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: tensor 0 has shape {ref_shape}, "
                f"tensor {i} has shape {t.shape}"
            )
        if t.device != ref_device:
            raise ValueError(
                f"Device mismatch: tensor 0 on {ref_device}, "
                f"tensor {i} on {t.device}"
            )


# ---------------------------------------------------------------------------
# 1. PSNR — Peak Signal-to-Noise Ratio  (Eq. 15-16)
# ---------------------------------------------------------------------------

def psnr(
    gt: torch.Tensor,
    pred: torch.Tensor,
    data_range: float = 2.0,
) -> float:
    """Compute PSNR between ground-truth and denoised images.

    Equation (15):  MSE = (1 / (M * N)) * sum_i sum_j (I_gt(i,j) - I_den(i,j))^2
    Equation (16):  PSNR = 10 * log10(I_max^2 / MSE)

    Args:
        gt:          Ground-truth tensor, shape [B, 1, H, W].
        pred:        Denoised (predicted) tensor, same shape as gt.
        data_range:  Maximum valid value (I_max) of the signal.  Default 2.0
                     for images normalised to [-1, 1].

    Returns:
        PSNR value in dB (float).  Returns +inf when pred == gt exactly.
    """
    _check_inputs(gt, pred)

    mse = F.mse_loss(pred, gt, reduction='mean')  # scalar over all elements
    if mse == 0.0:
        return float('inf')

    psnr_val = 10.0 * math.log10((data_range ** 2) / mse.item())
    return psnr_val


# ---------------------------------------------------------------------------
# 2. SSIM — Structural Similarity Index Measure  (Eq. 17)
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(
    size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """1-D Gaussian window for convolution."""
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return g


def _gaussian_window_2d(
    window_size: int = 11,
    sigma: float = 1.5,
    channels: int = 1,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a 2-D Gaussian window shaped [1, C, win, win]."""
    g1d = _gaussian_kernel_1d(window_size, sigma, device, dtype)
    g2d = g1d[:, None] * g1d[None, :]  # outer product
    window = g2d.view(1, 1, window_size, window_size).repeat(1, channels, 1, 1)
    return window


def ssim(
    gt: torch.Tensor,
    pred: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,
    K1: float = 0.01,
    K2: float = 0.03,
) -> float:
    """Compute SSIM between ground-truth and denoised images.

    Equation (17):
        SSIM(x,y) = [l(x,y)]^α * [c(x,y)]^β * [s(x,y)]^γ
    with α = β = γ = 1, using an 11×11 Gaussian window.

    Internally:
        l(x,y) = (2 μ_x μ_y + C1) / (μ_x^2 + μ_y^2 + C1)
        c(x,y) = (2 σ_x σ_y + C2) / (σ_x^2 + σ_y^2 + C2)
        s(x,y) = (σ_xy + C3)         / (σ_x σ_y + C3)
        C1 = (K1 * L)^2,  C2 = (K2 * L)^2,  C3 = C2 / 2

    Args:
        gt:           Ground-truth tensor, shape [B, 1, H, W].
        pred:         Denoised tensor, same shape.
        window_size:  Size of the Gaussian window (default 11).
        sigma:        Standard deviation of Gaussian kernel (default 1.5).
        data_range:   Dynamic range L (default 2.0 for [-1, 1]).
        K1, K2:       Small constants for numerical stability.

    Returns:
        Mean SSIM over batch (float in [0, 1]).  Higher is better.
    """
    _check_inputs(gt, pred)
    B, C, H, W = gt.shape

    L = data_range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    window = _gaussian_window_2d(
        window_size, sigma, C, device=gt.device, dtype=gt.dtype
    )

    # Local statistics via Gaussian filtering
    mu_x = F.conv2d(gt, window, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(pred, window, padding=window_size // 2, groups=C)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(gt * gt, window, padding=window_size // 2, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=C) - mu_y_sq
    sigma_xy = F.conv2d(gt * pred, window, padding=window_size // 2, groups=C) - mu_xy

    # Luminance, contrast, structure
    l_map = (2.0 * mu_xy + C1) / (mu_x_sq + mu_y_sq + C1)
    c_map = (2.0 * sigma_x_sq.sqrt() * sigma_y_sq.sqrt() + C2) / (sigma_x_sq + sigma_y_sq + C2)
    s_map = (sigma_xy + C2 / 2.0) / (sigma_x_sq.sqrt() * sigma_y_sq.sqrt() + C2 / 2.0)

    ssim_map = l_map * c_map * s_map
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# 3. CNR — Contrast-to-Noise Ratio  (Eq. 18)
# ---------------------------------------------------------------------------

def cnr(
    gt: torch.Tensor,
    pred: torch.Tensor,
    mask_A: torch.Tensor,
    mask_B: torch.Tensor,
) -> float:
    """Compute Contrast-to-Noise Ratio between two regions.

    Equation (18):
        CNR = |μ_A - μ_B| / sqrt(σ_A^2 + σ_B^2)

    where μ_A, σ_A^2 are the mean / variance of the denoised image within
    target region A (e.g. a lesion), and μ_B, σ_B^2 are those within a
    background region B (e.g. surrounding healthy tissue).

    Args:
        gt:      Ground-truth tensor, shape [B, 1, H, W].
        pred:    Denoised tensor, same shape.
        mask_A:  Binary mask for target region A, same shape (1 = inside A).
        mask_B:  Binary mask for background region B, same shape (1 = inside B).

    Returns:
        CNR value (float, higher is better).
    """
    _check_inputs(gt, pred, mask_A, mask_B)

    # Flatten over batch, channel, spatial dims
    pred_flat = pred.ravel()
    mask_A_flat = mask_A.ravel().bool()
    mask_B_flat = mask_B.ravel().bool()

    vals_A = pred_flat[mask_A_flat]
    vals_B = pred_flat[mask_B_flat]

    if vals_A.numel() == 0 or vals_B.numel() == 0:
        raise ValueError("One of the CNR masks contains no elements.")

    mu_A = vals_A.mean()
    mu_B = vals_B.mean()
    var_A = vals_A.var(unbiased=False)
    var_B = vals_B.var(unbiased=False)

    denominator = (var_A + var_B).sqrt()
    if denominator == 0.0:
        return 0.0

    cnr_val = (mu_A - mu_B).abs() / denominator
    return cnr_val.item()


# ---------------------------------------------------------------------------
# 4. ENL — Equivalent Number of Looks  (Eq. 19)
# ---------------------------------------------------------------------------

def enl(
    pred: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Compute Equivalent Number of Looks over a homogeneous ROI.

    Equation (19):
        ENL_i = μ_i^2 / σ_i^2

    where μ_i, σ_i^2 are the local mean and variance computed *within* the
    ROI of the denoised image.  A higher ENL indicates smoother homogeneous
    regions (less speckle noise).

    Args:
        pred:  Denoised tensor, shape [B, 1, H, W].
        mask:  Binary mask defining the homogeneous ROI, same shape.

    Returns:
        ENL value (float, higher is better).
    """
    _check_inputs(pred, mask)

    vals = pred[mask.bool()]
    if vals.numel() < 2:
        raise ValueError("ENL mask must contain at least 2 elements.")

    mu = vals.mean()
    var = vals.var(unbiased=False)
    if var == 0.0:
        return float('inf')  # perfectly homogeneous

    enl_val = (mu ** 2) / var
    return enl_val.item()


# ---------------------------------------------------------------------------
# 5. EPI — Edge Preservation Index  (Eq. 20)
# ---------------------------------------------------------------------------

def epi(
    noisy: torch.Tensor,
    denoised: torch.Tensor,
    edge_mask: torch.Tensor,
) -> float:
    """Compute Edge Preservation Index within an edge ROI.

    Equation (20):
        EPI = Σ_W Σ_H |I_d(x+1,y) - I_d(x,y)| / Σ_W Σ_H |I_n(x+1,y) - I_n(x,y)|

    where I_d is the denoised image, I_n is the noisy image, and the
    summations are carried out only over pixels inside the edge mask
    (and valid neighbours).  Horizontal differences are used by default;
    vertical differences are computed and averaged as well for a more
    robust measure.

    Args:
        noisy:      Noisy image tensor, shape [B, 1, H, W].
        denoised:   Denoised (predicted) tensor, same shape.
        edge_mask:  Binary edge-region mask, same shape.

    Returns:
        EPI value (float, higher is better; typical range ~ [0, >1]).
    """
    _check_inputs(noisy, denoised, edge_mask)

    # Horizontal differences (I[x+1, y] - I[x, y]) for x in [0, W-2]
    den_h = (denoised[:, :, :, 1:] - denoised[:, :, :, :-1]).abs()
    noi_h = (noisy[:, :, :, 1:] - noisy[:, :, :, :-1]).abs()

    # Only keep positions where BOTH pixels in the difference are inside the
    # edge mask.  Since the mask is binary, we intersect it with a one-pixel
    # right-shifted version along the width axis.
    edge_h = edge_mask[:, :, :, :-1] * edge_mask[:, :, :, 1:]

    # Vertical differences (I[x, y+1] - I[x, y]) for y in [0, H-2]
    den_v = (denoised[:, :, 1:, :] - denoised[:, :, :-1, :]).abs()
    noi_v = (noisy[:, :, 1:, :] - noisy[:, :, :-1, :]).abs()

    edge_v = edge_mask[:, :, :-1, :] * edge_mask[:, :, 1:, :]

    num = (den_h * edge_h).sum() + (den_v * edge_v).sum()
    den = (noi_h * edge_h).sum() + (noi_v * edge_v).sum()

    if den == 0.0:
        # All gradients are zero in the noisy image — EPI is undefined.
        # Return 1.0 as the identity (edges preserved exactly).
        return 1.0

    epi_val = num / den
    return epi_val.item()


# ---------------------------------------------------------------------------
# __main__ — quick smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("FAD-Net Evaluation Metrics — Smoke Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, C, H, W = 2, 1, 128, 128

    torch.manual_seed(42)

    # Simulate images in [-1, 1] range
    gt = 2.0 * torch.rand(B, C, H, W, device=device) - 1.0
    noisy = gt + 0.3 * torch.randn(B, C, H, W, device=device)
    pred = 0.9 * gt + 0.1 * noisy  # a reasonable "denoised" output

    # Dummy masks for CNR / ENL / EPI
    mask_A = torch.zeros(B, C, H, W, device=device)
    mask_B = torch.zeros(B, C, H, W, device=device)
    roi_mask = torch.zeros(B, C, H, W, device=device)
    edge_mask = torch.zeros(B, C, H, W, device=device)

    mask_A[:, :, 40:60, 40:60] = 1.0
    mask_B[:, :, 80:100, 80:100] = 1.0
    roi_mask[:, :, 20:80, 20:80] = 1.0
    edge_mask[:, :, 30:70, 30:70] = 1.0

    # ---- PSNR ----
    psnr_val = psnr(gt, pred, data_range=2.0)
    print(f"\n[PSNR]  shape={list(gt.shape)} →  {psnr_val:.4f} dB  (range: higher=better, +inf=perfect)")

    # ---- SSIM ----
    ssim_val = ssim(gt, pred, data_range=2.0)
    print(f"[SSIM]  shape={list(gt.shape)} →  {ssim_val:.6f}    (range: [0,1], higher=better)")

    # ---- CNR ----
    cnr_val = cnr(gt, pred, mask_A, mask_B)
    print(f"[CNR]   shape={list(gt.shape)} →  {cnr_val:.4f}      (range: [0,+inf), higher=better)")

    # ---- ENL ----
    enl_val = enl(pred, roi_mask)
    print(f"[ENL]   shape={list(gt.shape)} →  {enl_val:.4f}      (range: [0,+inf), higher=better)")

    # ---- EPI ----
    epi_val = epi(noisy, pred, edge_mask)
    print(f"[EPI]   shape={list(gt.shape)} →  {epi_val:.4f}      (range: [0,+inf), higher=better)")

    print("\n" + "=" * 60)
    print("All metrics executed successfully.")
    print("=" * 60)
