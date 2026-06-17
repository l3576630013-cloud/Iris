"""
FAD-Net SFE Pre-training — Contrastive Predictive Coding
==========================================================
Paper: FAD-Net, IEEE TIM 2026, Section III-B, Eq.(1)-(5), Fig.3

Stage 1 of the FAD-Net pipeline.  Trains the SFE (ResNet-18 backbone)
to encode structural features of OCT image patches via triplet
contrastive loss.  The pre-trained SFE weights are frozen during
Stage 2 (diffusion denoising).

Algorithm (Section III-B):
  1. Sample target patch  x_t (random P×P crop)
  2. Sample context patch x_c (spatially adjacent, offset ≤ P/2)
  3. Sample structurally similar patch x_s (long-range, texture-matched)
     via pixel-level cross-correlation (Eq.2-3)
  4. Extract features z_t, z_c, z_s via SFE encoder + global pool
  5. Triplet InfoNCE loss (Eq.4-5)

Hyperparameters (marked [SPECULATIVE] where paper is silent):
  P  = 32   patch size                    [SPECULATIVE]
  r  = 2    distance multiple (Eq.1)      [SPECULATIVE]
  tau = 0.07  temperature (Eq.5)          [SPECULATIVE]  (SimCLR standard)
  K  = 10   long-range candidate count    [SPECULATIVE]
  batch_size = 8                          [SPECULATIVE]
  lr  = 3e-4                              [SPECULATIVE]
  epochs = 50                             [SPECULATIVE]

Usage:
    python pretrain_sfe.py --data_dir D:/lyx/octa/OCTA(FULL) --epochs 50
"""

import os
import sys
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image as PILImage
from pathlib import Path
try:
    from tqdm import tqdm as _tqdm
    def tqdm_wrapper(iterable, **kw):
        return _tqdm(iterable, **kw)
    _has_tqdm = True
except ImportError:
    def tqdm_wrapper(iterable, **kw):
        return iterable
    _has_tqdm = False

# Import SFE backbone from existing module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sfe import SFE


# ============================================================
# Patch Sampling Utilities (Eq.1-3)
# ============================================================
def sample_target_patch(img: torch.Tensor, P: int):
    """
    Randomly crop a P×P target patch x_t from the image.

    Paper Section III-B1: "Randomly crop a P×P as the target patch."

    Args:
        img: [1, H, W]  single-channel OCT image
        P:   patch size

    Returns:
        x_t:  [1, P, P]
        (top, left): coordinates for downstream sampling
    """
    _, H, W = img.shape
    top = random.randint(0, H - P)
    left = random.randint(0, W - P)
    x_t = img[:, top:top + P, left:left + P]
    return x_t, (top, left)


def sample_context_patch(img: torch.Tensor, P: int, center: tuple):
    """
    Sample context patch x_c from the spatial neighbourhood of x_t.

    Paper Section III-B1:
        "Context Patch x_c: Sampled from the spatial adjacent area of
        x_t (offset ≤ P/2)"

    Args:
        img:    [1, H, W]
        P:      patch size
        center: (top, left) of target patch

    Returns:
        x_c: [1, P, P]
    """
    _, H, W = img.shape
    t0, l0 = center
    half = P // 2

    # Random offset within [-P/2, P/2] in both directions
    dy = random.randint(-half, half)
    dx = random.randint(-half, half)

    # Clamp to image bounds
    top = max(0, min(H - P, t0 + dy))
    left = max(0, min(W - P, l0 + dx))
    return img[:, top:top + P, left:left + P]


def pixel_cross_correlation(xt: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Pixel-level cross-correlation coefficient (Eq.2).

        Sim(X_t, X) = Cov(X_t, X) / (σ_Xt · σ_X)

    Computed as Pearson correlation over all pixel pairs.

    Args:
        xt: [1, P, P]  target patch
        x:  [1, P, P]  candidate patch

    Returns:
        sim: scalar correlation coefficient ∈ [-1, 1]
    """
    xt_flat = xt.flatten().float()
    x_flat = x.flatten().float()
    # Subtract mean (covariance numerator)
    xt_centered = xt_flat - xt_flat.mean()
    x_centered = x_flat - x_flat.mean()
    # Covariance
    cov = (xt_centered * x_centered).sum()
    # Standard deviations
    std_xt = xt_centered.norm()
    std_x = x_centered.norm()
    # Avoid division by zero
    denom = std_xt * std_x
    if denom < 1e-8:
        return torch.tensor(0.0)
    return cov / denom


def sample_similar_patch(img: torch.Tensor, P: int, center: tuple,
                         r: float = 2.0, K: int = 10) -> torch.Tensor:
    """
    Sample structurally similar patch x_s via long-range semantic matching
    (Eq.1 + Eq.3).

    Paper Section III-B2:
        1. Sample K long-range candidates with distance ≥ r·P from target
           (Eq.1: C = {P | dist(P, P_t) ≥ r·s})
        2. Compute pixel-level cross-correlation Sim(X_t, X) (Eq.2)
        3. Select argmax Sim(X_t, X) as x_s (Eq.3)

    Args:
        img:    [1, H, W]
        P:      patch size
        center: (top, left) of target patch
        r:      distance multiple (Eq.1).  Paper NOT specified → default 2.
        K:      number of candidate patches.  Paper NOT specified → default 10.

    Returns:
        x_s: [1, P, P]  structurally similar patch
    """
    _, H, W = img.shape
    t0, l0 = center
    min_dist = int(r * P)
    xt_patch = img[:, t0:t0 + P, l0:l0 + P]

    best_sim = -float('inf')
    best_patch = None
    attempts = 0
    max_attempts = K * 10  # avoid infinite loop on small images

    for _ in range(K):
        # Sample candidate far from target (Manhattan distance ≥ min_dist)
        found = False
        for _ in range(100):  # inner retry for valid placement
            top = random.randint(0, H - P)
            left = random.randint(0, W - P)
            dist = abs(top - t0) + abs(left - l0)  # Manhattan distance
            if dist >= min_dist:
                found = True
                break
            attempts += 1
            if attempts > max_attempts:
                break
        if not found:
            continue

        candidate = img[:, top:top + P, left:left + P]
        sim = pixel_cross_correlation(xt_patch, candidate)
        if sim > best_sim:
            best_sim = sim
            best_patch = candidate
        attempts += 1
        if attempts > max_attempts:
            break

    # Fallback: if no valid candidate found, use a random distant patch
    if best_patch is None:
        for _ in range(100):
            top = random.randint(0, H - P)
            left = random.randint(0, W - P)
            if abs(top - t0) + abs(left - l0) >= min_dist:
                best_patch = img[:, top:top + P, left:left + P]
                break
    if best_patch is None:
        # Last resort: any random patch
        top = random.randint(0, H - P)
        left = random.randint(0, W - P)
        best_patch = img[:, top:top + P, left:left + P]

    return best_patch


# ============================================================
# SFE Encoder with Pooling (for patch feature extraction)
# ============================================================
class SFEPatchEncoder(nn.Module):
    """
    Wraps the SFE backbone for patch-level feature extraction.

    Input:  [B, 1, P, P]   image patch
    Output: [B, D]          feature vector  (D = 512 for ResNet-18 layer4)

    The SFE (ResNet-18) has stride 32 total (stem=4, layer2=2, layer3=2,
    layer4=2).  For P=32 input, layer4 output is [B, 512, 1, 1], so
    spatial average yields [B, 512].
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.sfe = SFE(in_channels=in_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # → [B, C, 1, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, P, P]  image patch

        Returns:
            z: [B, 512]       feature vector
        """
        feats = self.sfe(x)              # dict: f1..f4
        z = self.pool(feats['f4'])       # [B, 512, 1, 1]
        z = z.view(z.shape[0], -1)       # [B, 512]
        z = F.normalize(z, dim=-1)       # L2-normalize (standard for InfoNCE)
        return z


# ============================================================
# Triplet InfoNCE Loss (Eq.4-5)
# ============================================================
def triplet_infonce_loss(z_t: torch.Tensor, z_c: torch.Tensor,
                         z_s: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    Triplet InfoNCE loss (Eq.4).

        L_con = (1 / 3N) * Σ_i [InfoNCE(z_t^i, {z_c^i, z_s^i})
                                + InfoNCE(z_c^i, {z_t^i, z_s^i})
                                + InfoNCE(z_s^i, {z_t^i, z_c^i})]

    Each InfoNCE term (Eq.5):
        InfoNCE(h_i, h_j) = -log( exp(sim(h_i, h_j)/τ)
                                 / Σ_k exp(sim(h_i, h_k)/τ) )

    where the denominator sums over the entire batch (all 3N vectors).

    Paper Section III-B3, Eq.(4)-(5).

    Args:
        z_t: [B, D]  target patch features
        z_c: [B, D]  context patch features
        z_s: [B, D]  similar patch features
        tau: temperature.  Paper NOT specified → default 0.07 (SimCLR).

    Returns:
        loss: scalar
    """
    B, D = z_t.shape

    # Concatenate all features: [3B, D]
    z_all = torch.cat([z_t, z_c, z_s], dim=0)  # [3B, D]

    # Cosine similarity matrix: [3B, 3B]
    sim = torch.matmul(z_all, z_all.T) / tau   # scaled by temperature

    # ---- Build positive-pair masks ----
    # For each anchor in z_t (indices 0..B-1), positives are:
    #   corresponding z_c (indices B..2B-1) and z_s (indices 2B..3B-1)
    # Similarly for z_c and z_s as anchors.
    mask = torch.zeros(3 * B, 3 * B, device=z_t.device)

    for i in range(B):
        # z_t[i] positives: z_c[i], z_s[i]
        mask[i, B + i] = 1.0         # z_t → z_c
        mask[i, 2 * B + i] = 1.0     # z_t → z_s
        # z_c[i] positives: z_t[i], z_s[i]
        mask[B + i, i] = 1.0         # z_c → z_t
        mask[B + i, 2 * B + i] = 1.0 # z_c → z_s
        # z_s[i] positives: z_t[i], z_c[i]
        mask[2 * B + i, i] = 1.0     # z_s → z_t
        mask[2 * B + i, B + i] = 1.0 # z_s → z_c

    # ---- InfoNCE: -log( pos_sum / all_sum ) ----
    # For numerical stability, subtract max per row
    sim_max = sim.max(dim=1, keepdim=True).values
    sim_exp = torch.exp(sim - sim_max)

    # Sum over positives and all
    pos_sum = (sim_exp * mask).sum(dim=1)          # [3B]
    all_sum = sim_exp.sum(dim=1)                    # [3B]

    # InfoNCE per sample (Eq.5)
    loss_per_sample = -torch.log(pos_sum / all_sum)  # [3B]

    # Mean over all 3B samples (Eq.4: 1/(3N) Σ ...)
    loss = loss_per_sample.sum() / (3 * B)

    return loss


# ============================================================
# Triplet Patch Dataset
# ============================================================
class TripletPatchDataset(Dataset):
    """
    Dataset that yields (x_t, x_c, x_s) triplet patches from OCT images.

    For each image, samples target/context/similar patches following
    the procedure in Section III-B.

    Args:
        data_dir:    path to directory of OCT images
        P:           patch size (default 32)
        r:           distance multiple for x_s candidates (default 2)
        K:           number of long-range candidates (default 10)
        target_size: image resize before patch sampling (default 512)
    """

    def __init__(self, data_dir: str, P: int = 32, r: float = 2.0,
                 K: int = 10, target_size: int = 512):
        self.data_dir = Path(data_dir)
        self.P = P
        self.r = r
        self.K = K
        self.target_size = target_size

        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
        self.paths = []
        for ext in exts:
            self.paths.extend(sorted(self.data_dir.rglob(ext)))
        self.paths = [str(p) for p in self.paths]

        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found in {data_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),  # [0, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Load and preprocess image
        img = PILImage.open(self.paths[idx]).convert('L')
        img_tensor = self.transform(img)  # [1, 512, 512]

        # Sample target patch
        x_t, center = sample_target_patch(img_tensor, self.P)

        # Sample context patch
        x_c = sample_context_patch(img_tensor, self.P, center)

        # Sample structurally similar patch
        x_s = sample_similar_patch(img_tensor, self.P, center,
                                    r=self.r, K=self.K)

        return x_t, x_c, x_s


# ============================================================
# Training Script
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='FAD-Net SFE Pre-training (CPC contrastive learning)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing OCT images')
    parser.add_argument('--output_dir', type=str, default='./output/sfe_pretrain',
                        help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size [SPECULATIVE: paper not specified]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs [SPECULATIVE: paper not specified]')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate [SPECULATIVE: paper not specified]')
    parser.add_argument('--P', type=int, default=32,
                        help='Patch size [SPECULATIVE: paper not specified]')
    parser.add_argument('--r', type=float, default=2.0,
                        help='Distance multiple for x_s (Eq.1) [SPECULATIVE]')
    parser.add_argument('--K', type=int, default=10,
                        help='Long-range candidate count (Eq.1) [SPECULATIVE]')
    parser.add_argument('--tau', type=float, default=0.07,
                        help='InfoNCE temperature (Eq.5) [SPECULATIVE]')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoader workers')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data:   {args.data_dir}")

    # ---- Dataset ----
    dataset = TripletPatchDataset(
        args.data_dir, P=args.P, r=args.r, K=args.K, target_size=512)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers,
                            drop_last=True, pin_memory=(device.type == 'cuda'))
    print(f"Images:  {len(dataset)}")
    print(f"Batches: {len(dataloader)} (batch_size={args.batch_size})")
    print(f"Patch:   {args.P}×{args.P}, r={args.r}, K={args.K}, tau={args.tau}")

    # ---- Model ----
    encoder = SFEPatchEncoder(in_channels=1).to(device)
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    # ---- Output ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training loop ----
    print(f"\n{'='*60}")
    print(f"Training SFE via CPC (Section III-B, Eq.4-5)")
    print(f"{'='*60}")

    best_loss = float('inf')
    for epoch in range(args.epochs):
        encoder.train()
        epoch_loss = 0.0
        pbar = tqdm_wrapper(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, (x_t, x_c, x_s) in enumerate(pbar):
            x_t = x_t.to(device)  # [B, 1, P, P]
            x_c = x_c.to(device)
            x_s = x_s.to(device)

            # Encode patches → feature vectors
            z_t = encoder(x_t)    # [B, 512]
            z_c = encoder(x_c)    # [B, 512]
            z_s = encoder(x_s)    # [B, 512]

            # Triplet InfoNCE loss (Eq.4-5)
            loss = triplet_infonce_loss(z_t, z_c, z_s, tau=args.tau)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if _has_tqdm:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            elif batch_idx % 10 == 0:
                print(f"  batch {batch_idx:4d}/{len(dataloader)}  loss={loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"  Epoch {epoch+1:3d}/{args.epochs}  avg_loss={avg_loss:.6f}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = output_dir / "sfe_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': encoder.sfe.state_dict(),
                'loss': avg_loss,
            }, str(ckpt_path))
            print(f"  → Saved best: {ckpt_path}")

        # Save periodic
        if (epoch + 1) % 10 == 0:
            ckpt_path = output_dir / f"sfe_epoch{epoch+1:03d}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': encoder.sfe.state_dict(),
                'loss': avg_loss,
            }, str(ckpt_path))

    print(f"\n{'='*60}")
    print(f"Pre-training complete. Best loss: {best_loss:.6f}")
    print(f"Checkpoint: {output_dir / 'sfe_best.pt'}")
    print(f"\nNext: train FAD-Net with --sfe_checkpoint {output_dir / 'sfe_best.pt'}")
    print(f"{'='*60}")


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Quick smoke test
        print("=" * 60)
        print("FAD-Net SFE Pre-training — Smoke Test")
        print("=" * 60)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        # Test 1: Patch sampling
        print(f"\n{'─'*60}")
        print("Test 1: Patch Sampling (Eq.1-3)")
        print(f"{'─'*60}")

        x = torch.randn(1, 512, 512)
        x_t, center = sample_target_patch(x, P=32)
        x_c = sample_context_patch(x, P=32, center=center)
        x_s = sample_similar_patch(x, P=32, center=center, r=2.0, K=10)

        print(f"  x_t: {list(x_t.shape)}  center={center}")
        print(f"  x_c: {list(x_c.shape)}")
        print(f"  x_s: {list(x_s.shape)}")
        print(f"  All same shape: {x_t.shape == x_c.shape == x_s.shape}")

        # Test 2: Cross-correlation
        print(f"\n{'─'*60}")
        print("Test 2: Pixel Cross-Correlation (Eq.2)")
        print(f"{'─'*60}")
        sim_self = pixel_cross_correlation(x_t, x_t)
        sim_rand = pixel_cross_correlation(x_t, torch.randn_like(x_t))
        print(f"  Sim(x_t, x_t):       {sim_self.item():.4f}  (expect ~1.0)")
        print(f"  Sim(x_t, random):    {sim_rand.item():.4f}  (expect ~0.0)")

        # Test 3: Encoder forward
        print(f"\n{'─'*60}")
        print("Test 3: SFE Encoder")
        print(f"{'─'*60}")
        encoder = SFEPatchEncoder(in_channels=1).to(device)
        B = 4
        batch = torch.randn(B, 1, 32, 32, device=device)
        with torch.no_grad():
            z = encoder(batch)
        print(f"  Input:  {list(batch.shape)}")
        print(f"  Output: {list(z.shape)}  (expect [{B}, 512])")
        # Verify: for P=32, ResNet-18 stride=32 → 1×1 spatial → pool → [B,512]
        print(f"  L2 norm per sample: {[f'{z[i].norm().item():.4f}' for i in range(B)]}")

        # Test 4: Loss
        print(f"\n{'─'*60}")
        print("Test 4: Triplet InfoNCE Loss (Eq.4-5)")
        print(f"{'─'*60}")
        z_t_batch = encoder(torch.randn(B, 1, 32, 32, device=device))
        z_c_batch = encoder(torch.randn(B, 1, 32, 32, device=device))
        z_s_batch = encoder(torch.randn(B, 1, 32, 32, device=device))
        loss = triplet_infonce_loss(z_t_batch, z_c_batch, z_s_batch, tau=0.07)
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Expected: ~ln(2*B) = {np.log(2*B):.4f} for random features")
        loss.backward()
        print(f"  Grad flow OK: {all(p.grad is not None for p in encoder.parameters() if p.requires_grad)}")

        print(f"\n{'='*60}")
        print("Smoke test passed.")
        print(f"Run with: python pretrain_sfe.py --data_dir D:/lyx/octa/OCTA(FULL) --epochs 50")
        print(f"{'='*60}")
