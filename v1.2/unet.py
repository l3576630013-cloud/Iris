"""
FAD-Net Noise Prediction U-Net (DDPM Standard 4-Level)
=========================================================
Paper: FAD-Net, IEEE TIM 2026, Section III-A, III-C3

Architecture: DDPM U-Net with 4 encoder levels + middle bottleneck.
              Timestep conditioning via sinusoidal embedding + MLP.

Input:  x_t     [B, 1, 512, 512]    noisy OCT image at timestep t
        t       [B]                  diffusion timestep (int)
Output: eps     [B, 1, 512, 512]    predicted noise epsilon

Encoder (4 levels):
  [B,  1, 512, 512] → conv_in → [B,  32, 512, 512]  (level 0, skip)
  → ResBlock + Down → [B,  64, 256, 256]  (level 1, skip)
  → ResBlock + Down → [B,  96, 128, 128]  (level 2, skip)
  → ResBlock + Down → [B, 128,  64,  64]  (level 3, skip)
  → ResBlock + Down → [B, 128,  32,  32]  (middle bottleneck)

Middle:
  ResBlock → Self-Attention → ResBlock

Decoder (4 levels, symmetric):
  Up + Concat(skip) + ResBlock → [B, 128,  64,  64]
  Up + Concat(skip) + ResBlock → [B,  96, 128, 128]
  Up + Concat(skip) + ResBlock → [B,  64, 256, 256]
  Up + Concat(skip) + ResBlock → [B,  32, 512, 512]
  conv_out → [B, 1, 512, 512]

Timestep embedding (DDPM standard):
  t → SinusoidalEmbedding(128) → Linear → SiLU → Linear → t_emb [B, 128]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sinusoidal Timestep Embedding
# ============================================================
class SinusoidalEmbedding(nn.Module):
    """
    DDPM standard sinusoidal position embedding for diffusion timesteps.

    t ∈ [0, T-1] → sin/cos embedding of dimension `dim`.

    Input:  t  [B] or [B, 1]   integer timestep
    Output: emb [B, dim]        sinusoidal embedding
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) *
                          torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer('freqs', freqs)  # [half]

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] or [B, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B, 1]
        args = t.float() * self.freqs  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
        return emb


# ============================================================
# ResBlock with Timestep Conditioning
# ============================================================
class ResBlock(nn.Module):
    """
    DDPM ResBlock: GroupNorm → SiLU → Conv3×3 → (time injection) →
                   GroupNorm → SiLU → Conv3×3 → (+ residual).

    Time embedding is projected via a Linear layer and added after
    the first convolution.

    Input:  x      [B, C_in,  H, W]
            t_emb  [B, D_t]
    Output: h      [B, C_out, H, W]
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int = 128,
                 num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_ch), in_ch)
        self.silu1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # Time embedding projection: D_t → out_ch
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )

        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.silu2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # Residual shortcut: 1×1 Conv if channel dim changes
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x:     [B, C_in,  H, W]
        # t_emb: [B, D_t]

        h = self.norm1(x)                                         # [B, C_in, H, W]
        h = self.silu1(h)                                         # [B, C_in, H, W]
        h = self.conv1(h)                                         # [B, C_out, H, W]

        # Time injection: project → [B, C_out, 1, 1] → add
        t_out = self.time_proj(t_emb)                             # [B, C_out]
        h = h + t_out[:, :, None, None]                           # broadcast

        h = self.norm2(h)                                         # [B, C_out, H, W]
        h = self.silu2(h)                                         # [B, C_out, H, W]
        h = self.conv2(h)                                         # [B, C_out, H, W]

        h = h + self.shortcut(x)                                  # residual

        return h


# ============================================================
# Self-Attention Block (used in middle bottleneck)
# ============================================================
class SelfAttention(nn.Module):
    """
    Self-attention on spatial feature map (used in middle block).

    Flattens H×W → sequence, applies multi-head attention,
    reshapes back to spatial. Head count = 4 (user spec).

    Input:  x [B, C, H, W]
    Output: x [B, C, H, W]   (same shape, with residual)
    """

    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.to_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        h = self.norm(x)                                          # [B, C, H, W]
        qkv = self.to_qkv(h)                                      # [B, 3*C, H, W]
        q, k, v = torch.chunk(qkv, 3, dim=1)                     # each [B, C, H, W]

        # Reshape to [B, n_heads, N, head_dim]
        q = q.view(B, self.n_heads, self.head_dim, H * W).transpose(-2, -1)
        k = k.view(B, self.n_heads, self.head_dim, H * W).transpose(-2, -1)
        v = v.view(B, self.n_heads, self.head_dim, H * W).transpose(-2, -1)
        # q: [B, nh, N, hd], k: [B, nh, N, hd], v: [B, nh, N, hd]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [B, nh, N, N]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)                                 # [B, nh, N, hd]
        out = out.transpose(-2, -1).contiguous().view(B, C, H, W)  # [B, C, H, W]
        out = self.to_out(out)                                      # [B, C, H, W]

        return out + x  # residual


# ============================================================
# Downsample / Upsample
# ============================================================
class Downsample(nn.Module):
    """2x downsampling via stride-2 Conv3×3."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3,
                              stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → [B, C, H/2, W/2]
        return self.conv(x)


class Upsample(nn.Module):
    """2x upsampling via nearest-neighbor + Conv3×3."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3,
                              padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → [B, C, 2H, 2W]
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


# ============================================================
# DDPM U-Net (4-Level)
# ============================================================
class UNet(nn.Module):
    """
    DDPM-standard 4-level U-Net for noise prediction.

    Encoder channels: [32, 64, 96, 128]  (user specified)
    Decoder: symmetric
    Middle: ResBlock → SelfAttention → ResBlock
    Skip connections between encoder and decoder at each level.

    Compatible with DDPM diffusion pipeline:
        Forward:  epsilon = unet(x_t, t)
        Reverse:  x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-bar_alpha_t) * epsilon)

    Input:  x_t [B, 1, 512, 512]   noisy OCT image
            t   [B]                 diffusion timestep
    Output: eps [B, 1, 512, 512]   predicted noise
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 enc_channels: list = None, time_emb_dim: int = 128,
                 num_res_blocks: int = 2):
        """
        Args:
            in_channels:     input channels (1 for grayscale OCT)
            out_channels:    output channels (1 = predicted noise per pixel)
            enc_channels:    encoder channel list. Default [32, 64, 96, 128].
                             Source: USER SPECIFIED.
            time_emb_dim:    timestep embedding dimension. Default 128.
                             Source: USER SPECIFIED.
            num_res_blocks:  number of ResBlocks per level. Default 2.
                             Paper NOT specified; DDPM standard.
        """
        super().__init__()
        if enc_channels is None:
            enc_channels = [32, 64, 96, 128]

        self.enc_channels = enc_channels
        self.time_emb_dim = time_emb_dim
        ch0, ch1, ch2, ch3 = enc_channels

        # ---- Timestep embedding ----
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_emb_dim),        # t → [B, 128]
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ---- Input projection ----
        # [B, 1, 512, 512] → [B, 32, 512, 512]
        self.conv_in = nn.Conv2d(in_channels, ch0, kernel_size=3, padding=1)

        # ============================================================
        # Encoder
        # ============================================================

        # ---- Level 0 (512²): ch0 ----
        self.enc0 = nn.ModuleList([
            ResBlock(ch0, ch0, time_emb_dim) for _ in range(num_res_blocks)
        ])
        self.down0 = Downsample(ch0)  # 512 → 256

        # ---- Level 1 (256²): ch1 ----
        self.enc1 = nn.ModuleList([
            ResBlock(ch0, ch1, time_emb_dim) if i == 0 else ResBlock(ch1, ch1, time_emb_dim)
            for i in range(num_res_blocks)
        ])
        self.down1 = Downsample(ch1)  # 256 → 128

        # ---- Level 2 (128²): ch2 ----
        self.enc2 = nn.ModuleList([
            ResBlock(ch1, ch2, time_emb_dim) if i == 0 else ResBlock(ch2, ch2, time_emb_dim)
            for i in range(num_res_blocks)
        ])
        self.down2 = Downsample(ch2)  # 128 → 64

        # ---- Level 3 (64²): ch3 ----
        self.enc3 = nn.ModuleList([
            ResBlock(ch2, ch3, time_emb_dim) if i == 0 else ResBlock(ch3, ch3, time_emb_dim)
            for i in range(num_res_blocks)
        ])
        self.down3 = Downsample(ch3)  # 64 → 32

        # ============================================================
        # Middle (32²)
        # ============================================================
        self.mid_res1 = ResBlock(ch3, ch3, time_emb_dim)
        self.mid_attn = SelfAttention(ch3, n_heads=4)
        self.mid_res2 = ResBlock(ch3, ch3, time_emb_dim)

        # ============================================================
        # Decoder (symmetric)
        # ============================================================

        # ---- Level 3' (32² → 64²): ch3 ----
        self.up3 = Upsample(ch3)  # 32 → 64
        # After concat with enc3 skip: ch3 + ch3 = 2*ch3 → ch3 → ch2
        self.dec3 = nn.ModuleList([
            ResBlock(ch3 + ch3, ch3, time_emb_dim) if i == 0 else ResBlock(ch3, ch3, time_emb_dim)
            for i in range(num_res_blocks)
        ])

        # ---- Level 2' (64² → 128²): ch3 → ch2 ----
        self.up2 = Upsample(ch3)  # 64 → 128
        self.dec2 = nn.ModuleList([
            ResBlock(ch3 + ch2, ch2, time_emb_dim) if i == 0 else ResBlock(ch2, ch2, time_emb_dim)
            for i in range(num_res_blocks)
        ])

        # ---- Level 1' (128² → 256²): ch2 → ch1 ----
        self.up1 = Upsample(ch2)  # 128 → 256
        self.dec1 = nn.ModuleList([
            ResBlock(ch2 + ch1, ch1, time_emb_dim) if i == 0 else ResBlock(ch1, ch1, time_emb_dim)
            for i in range(num_res_blocks)
        ])

        # ---- Level 0' (256² → 512²): ch1 → ch0 ----
        self.up0 = Upsample(ch1)  # 256 → 512
        self.dec0 = nn.ModuleList([
            ResBlock(ch1 + ch0, ch0, time_emb_dim) if i == 0 else ResBlock(ch0, ch0, time_emb_dim)
            for i in range(num_res_blocks)
        ])

        # ---- Output projection ----
        # [B, 32, 512, 512] → [B, 1, 512, 512]
        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(8, ch0), ch0),
            nn.SiLU(),
            nn.Conv2d(ch0, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, 512, 512]  noisy OCT image at timestep t
            t: [B]                diffusion timestep (int, e.g. 0..999)

        Returns:
            eps: [B, 1, 512, 512]  predicted noise
        """
        # ---- Timestep embedding ----
        t_emb = self.time_embed(t)  # [B, 128]

        # ---- Input projection ----
        h = self.conv_in(x)  # [B, 32, 512, 512]

        # ============================================================
        # Encoder (downward path, store skips)
        # ============================================================

        # Level 0 (512²)
        for block in self.enc0:
            h = block(h, t_emb)  # [B, 32, 512, 512]
        skip0 = h
        h = self.down0(h)  # [B, 32, 256, 256]

        # Level 1 (256²)
        for block in self.enc1:
            h = block(h, t_emb)  # [B, 64, 256, 256]
        skip1 = h
        h = self.down1(h)  # [B, 64, 128, 128]

        # Level 2 (128²)
        for block in self.enc2:
            h = block(h, t_emb)  # [B, 96, 128, 128]
        skip2 = h
        h = self.down2(h)  # [B, 96, 64, 64]

        # Level 3 (64²)
        for block in self.enc3:
            h = block(h, t_emb)  # [B, 128, 64, 64]
        skip3 = h
        h = self.down3(h)  # [B, 128, 32, 32]

        # ============================================================
        # Middle (32² bottleneck)
        # ============================================================
        h = self.mid_res1(h, t_emb)  # [B, 128, 32, 32]
        h = self.mid_attn(h)          # [B, 128, 32, 32]
        h = self.mid_res2(h, t_emb)  # [B, 128, 32, 32]

        # ============================================================
        # Decoder (upward path, concat skips)
        # ============================================================

        # Level 3' (32² → 64²)
        h = self.up3(h)                    # [B, 128, 64, 64]
        h = torch.cat([h, skip3], dim=1)  # [B, 256, 64, 64]
        for block in self.dec3:
            h = block(h, t_emb)            # [B, 128, 64, 64]

        # Level 2' (64² → 128²)
        h = self.up2(h)                    # [B, 128, 128, 128]
        h = torch.cat([h, skip2], dim=1)  # [B, 224, 128, 128]  (128+96)
        for block in self.dec2:
            h = block(h, t_emb)            # [B, 96, 128, 128]

        # Level 1' (128² → 256²)
        h = self.up1(h)                    # [B, 96, 256, 256]
        h = torch.cat([h, skip1], dim=1)  # [B, 160, 256, 256]  (96+64)
        for block in self.dec1:
            h = block(h, t_emb)            # [B, 64, 256, 256]

        # Level 0' (256² → 512²)
        h = self.up0(h)                    # [B, 64, 512, 512]
        h = torch.cat([h, skip0], dim=1)  # [B, 96, 512, 512]  (64+32)
        for block in self.dec0:
            h = block(h, t_emb)            # [B, 32, 512, 512]

        # ---- Output projection ----
        eps = self.conv_out(h)  # [B, 1, 512, 512]

        return eps


# ============================================================
# Test Code
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FAD-Net U-Net (DDPM Standard) — Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ---- Instantiate ----
    unet = UNet(
        in_channels=1,
        out_channels=1,
        enc_channels=[32, 64, 96, 128],
        time_emb_dim=128,
        num_res_blocks=2,
    ).to(device)

    n_params = sum(p.numel() for p in unet.parameters())
    print(f"\nTotal parameters: {n_params:,}")

    # ---- Test 1: Forward pass ----
    print(f"\n{'─' * 60}")
    print("Test 1: Forward Pass")
    print(f"{'─' * 60}")

    B = 2
    x_t = torch.randn(B, 1, 512, 512, device=device)
    t = torch.randint(0, 1000, (B,), device=device)

    print(f"  x_t:  {list(x_t.shape)}")
    print(f"  t:    {t.tolist()}")

    with torch.no_grad():
        eps = unet(x_t, t)

    print(f"  eps:  {list(eps.shape)}")
    print(f"  Shape match input: {eps.shape == x_t.shape}")
    print(f"  Value range: [{eps.min().item():.4f}, {eps.max().item():.4f}]")

    # ---- Test 2: Timestep sensitivity ----
    print(f"\n{'─' * 60}")
    print("Test 2: Timestep Sensitivity")
    print(f"{'─' * 60}")

    x_test = torch.randn(1, 1, 512, 512, device=device)
    eps_list = []
    for ts in [0, 250, 500, 750, 999]:
        with torch.no_grad():
            e = unet(x_test, torch.tensor([ts], device=device))
        eps_list.append(e)
        print(f"  t={ts:3d}: eps mean={e.mean().item():.4f}, std={e.std().item():.4f}")

    # Check: different t should give DIFFERENT predictions
    all_same = all(torch.allclose(eps_list[0], e, atol=1e-3) for e in eps_list[1:])
    print(f"  All timesteps produce identical output: {all_same}  "
          f"(should be False — UNet is timestep-conditioned)")

    # ---- Test 3: Memory footprint ----
    print(f"\n{'─' * 60}")
    print("Test 3: Memory Footprint (Estimated)")
    print(f"{'─' * 60}")

    if device.type == 'cuda':
        mem = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"  Peak GPU memory: {mem:.1f} MB")
    else:
        print(f"  Running on CPU — no GPU memory stats.")

    # ---- Test 4: Gradient flow ----
    print(f"\n{'─' * 60}")
    print("Test 4: Gradient Flow")
    print(f"{'─' * 60}")

    unet_grad = UNet(enc_channels=[32, 64, 96, 128], time_emb_dim=128).to(device)
    x_grad = torch.randn(1, 1, 512, 512, device=device, requires_grad=True)
    t_grad = torch.tensor([500], device=device)

    eps_grad = unet_grad(x_grad, t_grad)
    loss = eps_grad.sum()
    loss.backward()

    print(f"  x.grad shape: {list(x_grad.grad.shape)}")
    print(f"  x.grad max:   {x_grad.grad.abs().max().item():.6f}")
    print(f"  x.grad zero:  {(x_grad.grad == 0).all().item()}")
    print(f"  Grad flow OK: {x_grad.grad is not None and x_grad.grad.abs().max() > 0}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("U-Net Module Summary")
    print(f"{'=' * 60}")
    print(f"""
    Architecture:   DDPM-standard 4-level U-Net
    Parameters:     {n_params:,}

    Encoder:
      conv_in:        [B,  1, 512, 512] → [B,  32, 512, 512]
      Level 0:  2×ResBlock                   [B,  32, 512, 512]  (skip)
      Down0:                                  [B,  32, 256, 256]
      Level 1:  2×ResBlock                   [B,  64, 256, 256]  (skip)
      Down1:                                  [B,  64, 128, 128]
      Level 2:  2×ResBlock                   [B,  96, 128, 128]  (skip)
      Down2:                                  [B,  96,  64,  64]
      Level 3:  2×ResBlock                   [B, 128,  64,  64]  (skip)
      Down3:                                  [B, 128,  32,  32]

    Middle (32x32):
      ResBlock → SelfAttention(4 heads) → ResBlock  [B, 128, 32, 32]

    Decoder:
      Up3 + cat(skip3):  [B, 128+128, 64, 64]  → 2×ResBlock → [B, 128, 64, 64]
      Up2 + cat(skip2):  [B, 128+96, 128, 128]  → 2×ResBlock → [B,  96, 128, 128]
      Up1 + cat(skip1):  [B,  96+64, 256, 256]  → 2×ResBlock → [B,  64, 256, 256]
      Up0 + cat(skip0):  [B,  64+32, 512, 512]  → 2×ResBlock → [B,  32, 512, 512]
      conv_out:                                        → [B, 1, 512, 512]

    Timestep embedding:
      t [B] → SinusoidalEmbedding(128) → Linear → SiLU → Linear → [B, 128]

    Integration with FAD-Net:
      This U-Net outputs noise prediction epsilon.
      For F_fuse injection (TAWG + MHFB + SFE), a separate wrapper
      will prepend the F_fuse features into decoder ResBlocks.
      (Eq.9: F_fuse = (1-W_t)*F_enc + W_t*F_h)

    Speculative / User-specified:
      enc_channels = [32, 64, 96, 128]   USER SPECIFIED
      time_emb_dim = 128                 USER SPECIFIED
      num_res_blocks = 2                 DDPM standard
      GroupNorm groups = 8               DDPM standard
      Self-attention at bottleneck only  DDPM standard
    """)
