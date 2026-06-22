"""
FAD-Net evaluation package.

Metrics from the paper:
  "FAD-Net: Frequency-Aware Decomposition Network for Medical Image Denoising"
  IEEE TIM 2026, Section IV-B, Equations (15)-(20).

Available functions:
    psnr   — Peak Signal-to-Noise Ratio           (Eq. 15-16)
    ssim   — Structural Similarity Index Measure   (Eq. 17)
    cnr    — Contrast-to-Noise Ratio               (Eq. 18)
    enl    — Equivalent Number of Looks            (Eq. 19)
    epi    — Edge Preservation Index               (Eq. 20)
"""

from .metrics import cnr, enl, epi, psnr, ssim

__all__ = ['psnr', 'ssim', 'cnr', 'enl', 'epi']
