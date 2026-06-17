"""
FAD-Net DWT Module — Haar Wavelet Frequency Decomposition
===========================================================
Paper: FAD-Net: Unsupervised Frequency-Aware Diffusion Network
       for OCT Speckle Reduction (IEEE TIM 2026)

Reference: Section III-C1, Eq.(6)

    I_LL, I_LH, I_HL, I_HH = DWT(I)

Architecture:
    Input  → Row-wise Haar → Column-wise Haar → 4 Subbands
    I [B,C,H,W] → I_LL, I_LH, I_HL, I_HH  each [B, C, H/2, W/2]

The module supports:
    - Orthogonal Haar wavelet (1/sqrt(2) norm)
    - Forward decomposition (DWT)
    - Inverse reconstruction (IDWT) for verification
    - Multi-level decomposition
    - Per-channel decomposition via grouped convolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT(nn.Module):
    """
    Haar Discrete Wavelet Transform for 2D images.

    Decomposes input into 4 subbands:
        I_LL : low-frequency approximation (global structure)
        I_LH : high-frequency horizontal detail (vertical edges)
        I_HL : high-frequency vertical detail (horizontal edges)
        I_HH : high-frequency diagonal detail (corners / texture)

    Paper mapping (Eq.6):
        I_LL = I_LL   (approximation coefficients)
        I_LH = I_LH   (horizontal details)
        I_HL = I_HL   (vertical details)
        I_HH = I_HH   (diagonal details)

    Input:  [B, C, H, W]
    Output: (I_LL, I_LH, I_HL, I_HH)  each [B, C, H/2, W/2]
    """

    def __init__(self):
        super().__init__()
        # Haar wavelet filters (orthogonal normalization: 1/sqrt(2))
        # Low-pass:  [1,  1] / sqrt(2)
        # High-pass: [1, -1] / sqrt(2)
        value = 1.0 / (2.0 ** 0.5)
        lo = torch.tensor([value, value])   # [2]  low-pass
        hi = torch.tensor([value, -value])  # [2]  high-pass

        # Register as buffer so they move with .to(device)
        self.register_buffer('lo', lo)
        self.register_buffer('hi', hi)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Noisy OCT image  [B, C, H, W]

        Returns:
            I_LL: Low-frequency approximation   [B, C, H/2, W/2]
            I_LH: Horizontal high-freq detail    [B, C, H/2, W/2]
            I_HL: Vertical high-freq detail      [B, C, H/2, W/2]
            I_HH: Diagonal high-freq detail      [B, C, H/2, W/2]
        """
        B, C, H, W = x.shape

        # Check even dimensions (required by DWT downsampling)
        assert H % 2 == 0 and W % 2 == 0, \
            f"DWT requires even H,W. Got H={H}, W={W}."

        # ----- Stage 1: Row-wise (horizontal) filtering -----
        # Reshape filters to [out_ch, in_ch/groups, kernel_H, kernel_W]
        lo_kernel = self.lo.view(1, 1, 1, 2)  # [1, 1, 1, 2]
        hi_kernel = self.hi.view(1, 1, 1, 2)  # [1, 1, 1, 2]

        # Repeat for each input channel (groups=C → per-channel conv)
        lo_kernel = lo_kernel.repeat(C, 1, 1, 1)  # [C, 1, 1, 2]
        hi_kernel = hi_kernel.repeat(C, 1, 1, 1)  # [C, 1, 1, 2]

        # Row-wise: stride=(1,2) → downsample W by 2
        L = F.conv2d(x, lo_kernel, stride=(1, 2), groups=C, padding=0)  # [B,C,H,W/2]
        H = F.conv2d(x, hi_kernel, stride=(1, 2), groups=C, padding=0)  # [B,C,H,W/2]

        # ----- Stage 2: Column-wise (vertical) filtering -----
        lo_kernel_v = self.lo.view(1, 1, 2, 1)  # [1, 1, 2, 1]
        hi_kernel_v = self.hi.view(1, 1, 2, 1)  # [1, 1, 2, 1]
        lo_kernel_v = lo_kernel_v.repeat(C, 1, 1, 1)  # [C, 1, 2, 1]
        hi_kernel_v = hi_kernel_v.repeat(C, 1, 1, 1)  # [C, 1, 2, 1]

        # Column-wise on L: stride=(2,1) → downsample H by 2
        I_LL = F.conv2d(L, lo_kernel_v, stride=(2, 1), groups=C, padding=0)  # [B,C,H/2,W/2]
        I_LH = F.conv2d(L, hi_kernel_v, stride=(2, 1), groups=C, padding=0)  # [B,C,H/2,W/2]

        # Column-wise on H: stride=(2,1) → downsample H by 2
        I_HL = F.conv2d(H, lo_kernel_v, stride=(2, 1), groups=C, padding=0)  # [B,C,H/2,W/2]
        I_HH = F.conv2d(H, hi_kernel_v, stride=(2, 1), groups=C, padding=0)  # [B,C,H/2,W/2]

        return I_LL, I_LH, I_HL, I_HH


class HaarIDWT(nn.Module):
    """
    Inverse Haar Discrete Wavelet Transform.

    Uses manual zero-insertion + symmetric padding + regular conv2d
    (NOT conv_transpose2d which has output-offset issues with
    kernel_size == stride == 2).

    Standard inverse DWT algorithm:
        1. Zero-insert along H (upsample columns by 2)
        2. Filter columns with reconstruction filters, sum LL + LH (or HL + HH)
        3. Zero-insert along W (upsample rows by 2)
        4. Filter rows with reconstruction filters, sum L + H

    The F.pad(..., (0,0,1,0)) on H and F.pad(..., (1,0,0,0)) on W compensates
    for the kernel-induced shift, ensuring output size == 2 * input size.

    Input:  (I_LL, I_LH, I_HL, I_HH) each [B, C, H/2, W/2]
    Output: I_reconstructed  [B, C, H, W]
    """

    def __init__(self):
        super().__init__()
        # Inverse Haar filters
        # Forward: lo=[1,1]/√2, hi=[1,-1]/√2
        # Inverse: lo_inv=[1,1]/√2 (same), hi_inv=[-1,1]/√2 (transposed)
        value = 1.0 / (2.0 ** 0.5)
        lo_inv = torch.tensor([value, value])    # [2]
        hi_inv = torch.tensor([-value, value])   # [2]

        self.register_buffer('lo_inv', lo_inv)
        self.register_buffer('hi_inv', hi_inv)

    def forward(self, I_LL, I_LH, I_HL, I_HH):
        """
        Args:
            I_LL, I_LH, I_HL, I_HH: each [B, C, H_half, W_half]

        Returns:
            I: reconstructed image [B, C, 2*H_half, 2*W_half]
        """
        assert I_LL.shape == I_LH.shape == I_HL.shape == I_HH.shape
        B, C, H_half, W_half = I_LL.shape

        # Reshape filters: [C, 1, kernel_H, kernel_W]
        # Column filter (operates along H)
        lo_v = self.lo_inv.view(1, 1, 2, 1).expand(C, -1, -1, -1)
        hi_v = self.hi_inv.view(1, 1, 2, 1).expand(C, -1, -1, -1)
        # Row filter (operates along W)
        lo_h = self.lo_inv.view(1, 1, 1, 2).expand(C, -1, -1, -1)
        hi_h = self.hi_inv.view(1, 1, 1, 2).expand(C, -1, -1, -1)

        # ----- Stage 1: Column-wise inverse (upsample H by 2) -----
        # Step A: zero-insertion along H dimension
        LL_up = _zero_insert_h(I_LL)   # [B, C, 2*H_h, W_h]
        LH_up = _zero_insert_h(I_LH)
        HL_up = _zero_insert_h(I_HL)
        HH_up = _zero_insert_h(I_HH)

        # Step B: pad H-dim top by 1, then filter along H
        # pad format: (pad_left, pad_right, pad_top, pad_bottom)
        L = (F.conv2d(F.pad(LL_up, (0, 0, 1, 0)), lo_v, groups=C) +
             F.conv2d(F.pad(LH_up, (0, 0, 1, 0)), hi_v, groups=C))  # [B, C, 2*H_h, W_h]
        H = (F.conv2d(F.pad(HL_up, (0, 0, 1, 0)), lo_v, groups=C) +
             F.conv2d(F.pad(HH_up, (0, 0, 1, 0)), hi_v, groups=C))  # [B, C, 2*H_h, W_h]

        # ----- Stage 2: Row-wise inverse (upsample W by 2) -----
        L_up = _zero_insert_w(L)   # [B, C, 2*H_h, 2*W_h]
        H_up = _zero_insert_w(H)

        # pad W-dim left by 1, then filter along W
        I = (F.conv2d(F.pad(L_up, (1, 0, 0, 0)), lo_h, groups=C) +
             F.conv2d(F.pad(H_up, (1, 0, 0, 0)), hi_h, groups=C))  # [B, C, 2*H_h, 2*W_h]

        return I


def _zero_insert_h(x: torch.Tensor) -> torch.Tensor:
    """Insert zeros along H dimension: [B,C,H,W] → [B,C,2H,W]"""
    B, C, H, W = x.shape
    up = torch.zeros(B, C, 2 * H, W, device=x.device, dtype=x.dtype)
    up[:, :, 0::2, :] = x
    return up


def _zero_insert_w(x: torch.Tensor) -> torch.Tensor:
    """Insert zeros along W dimension: [B,C,H,W] → [B,C,H,2W]"""
    B, C, H, W = x.shape
    up = torch.zeros(B, C, H, 2 * W, device=x.device, dtype=x.dtype)
    up[:, :, :, 0::2] = x
    return up


def get_high_frequency_subbands(I_LH, I_HL, I_HH):
    """
    Concatenate the 3 high-frequency subbands into a single tensor
    for MHFB input.

    Paper: MHFB receives {I_LH, I_HL, I_HH} as the high-frequency detail.

    Args:
        I_LH: [B, C, H/2, W/2]
        I_HL: [B, C, H/2, W/2]
        I_HH: [B, C, H/2, W/2]

    Returns:
        HF: [B, 3*C, H/2, W/2]  — concatenated along channel dim
    """
    return torch.cat([I_LH, I_HL, I_HH], dim=1)


# ============================================================
# Test Code
# ============================================================
if __name__ == "__main__":
    import os
    import numpy as np
    from PIL import Image as PILImage

    print("=" * 60)
    print("FAD-Net DWT Module — Test")
    print("=" * 60)

    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ================================================================
    # CONFIG: Set your OCTA image paths here
    # Images will be resized to 512×512, normalized to [0, 1], grayscale
    # ================================================================
    OCTA_IMAGE_PATHS = [
        # TODO: replace with your actual OCTA image paths
        # "C:/path/to/your/octa_image_1.png",
        # "C:/path/to/your/octa_image_2.png",
    ]

    if not OCTA_IMAGE_PATHS:
        print("\n[WARNING] No OCTA image paths configured.")
        print("  Edit OCTA_IMAGE_PATHS in the __main__ block to point to your images.")
        print("  Using random tensor as fallback for shape verification only.\n")

        # Fallback: random tensor (shape verification only)
        img_tensor = torch.randn(2, 1, 512, 512).to(device)
        print(f"  [FALLBACK] Random input shape: {list(img_tensor.shape)}")
    else:
        print(f"\n  Loading {len(OCTA_IMAGE_PATHS)} OCTA image(s)...")
        imgs = []
        for p in OCTA_IMAGE_PATHS:
            img = PILImage.open(p).convert('L')        # grayscale
            img = img.resize((512, 512), PILImage.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            imgs.append(arr)
            print(f"    {os.path.basename(p)}: {arr.shape}, range [{arr.min():.4f}, {arr.max():.4f}]")
        img_np = np.stack(imgs, axis=0)                # [B, 512, 512]
        img_tensor = torch.from_numpy(img_np).unsqueeze(1).to(device)  # [B, 1, 512, 512]
        print(f"  Input batch shape: {list(img_tensor.shape)}")

    # --- DWT Forward ---
    dwt = HaarDWT().to(device)
    print(f"\n{'─' * 60}")
    print("HaarDWT Forward")
    print(f"{'─' * 60}")

    with torch.no_grad():
        I_LL, I_LH, I_HL, I_HH = dwt(img_tensor)

    B = img_tensor.shape[0]
    print(f"\n  I_LL (low-freq approximation):  {list(I_LL.shape)}")
    print(f"  I_LH (horizontal detail):       {list(I_LH.shape)}")
    print(f"  I_HL (vertical detail):         {list(I_HL.shape)}")
    print(f"  I_HH (diagonal detail):         {list(I_HH.shape)}")

    for b in range(B):
        print(f"\n  Sample {b+1} value ranges:")
        print(f"    I_LL: [{I_LL[b].min().item():.6f}, {I_LL[b].max().item():.6f}]")
        print(f"    I_LH: [{I_LH[b].min().item():.6f}, {I_LH[b].max().item():.6f}]")
        print(f"    I_HL: [{I_HL[b].min().item():.6f}, {I_HL[b].max().item():.6f}]")
        print(f"    I_HH: [{I_HH[b].min().item():.6f}, {I_HH[b].max().item():.6f}]")

    # --- Concatenated HF subbands (for MHFB) ---
    HF = torch.cat([I_LH, I_HL, I_HH], dim=1)
    C = img_tensor.shape[1]
    print(f"\n  HF concat (MHFB input):  {list(HF.shape)}")
    print(f"    = cat([I_LH, I_HL, I_HH], dim=1)  ← [B, {3*C}, H/2, W/2]")

    # --- Inverse DWT (reconstruction verification) ---
    idwt = HaarIDWT().to(device)
    print(f"\n{'─' * 60}")
    print("HaarIDWT — Reconstruction Verification")
    print(f"{'─' * 60}")

    with torch.no_grad():
        I_recon = idwt(I_LL, I_LH, I_HL, I_HH)

    recon_error = (I_recon - img_tensor).abs()
    print(f"\n  Reconstructed shape:  {list(I_recon.shape)}")
    print(f"  Max abs error:        {recon_error.max().item():.8f}")
    print(f"  Mean abs error:       {recon_error.mean().item():.8f}")

    # --- Multi-channel test ---
    print(f"\n{'─' * 60}")
    print("Multi-channel Test (C=3)")
    print(f"{'─' * 60}")
    x3 = torch.randn(2, 3, 512, 512).to(device)
    LL3, LH3, HL3, HH3 = dwt(x3)
    print(f"  Input:  {list(x3.shape)}")
    print(f"  I_LL:   {list(LL3.shape)}")
    print(f"  I_LH:   {list(LH3.shape)}")
    print(f"  I_HL:   {list(HL3.shape)}")
    print(f"  I_HH:   {list(HH3.shape)}")
    HF3 = torch.cat([LH3, HL3, HH3], dim=1)
    print(f"  HF cat: {list(HF3.shape)}  ← [B, 9, H/2, W/2]")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("Module Summary")
    print(f"{'=' * 60}")
    print(f"""
    HaarDWT:
        Input:  [B, C, H, W]
        Output: I_LL [B, C, H/2, W/2]   ← low-freq, not used in MHFB
                I_LH [B, C, H/2, W/2]   ← horizontal detail
                I_HL [B, C, H/2, W/2]   ← vertical detail
                I_HH [B, C, H/2, W/2]   ← diagonal detail

    MHFB receives (per paper Section III-C2):
        HF = cat([I_LH, I_HL, I_HH], dim=1) → [B, 3*C, H/2, W/2]

    Paper hyperparameters (all from Section IV-A2):
        H = 512, W = 512    (image resize)
        C = 1               (OCT grayscale; RGB → C=3)

    To run with real images:
        Edit OCTA_IMAGE_PATHS in the __main__ block.
    """)
