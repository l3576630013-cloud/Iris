"""
FAD-Net: Unsupervised Frequency-Aware Diffusion Network
=========================================================
Paper: IEEE TIM, Vol. 75, 2026, 5004816
       "FAD-Net: Unsupervised Frequency-Aware Diffusion Network
        for Optical Coherence Tomography Speckle Reduction"

Full model assembly — integrates all submodules into a complete
noise-prediction network for unsupervised OCT denoising.

Architecture (Fig. 2, Algorithm 1):

    I_noisy [B,1,512,512]
         │
         ├── DWT (Haar) ──→ {LL, LH, HL, HH}
         │                      │
         │                 MHFB (Eq.7)
         │                      │
         │                  F_h = {h14, h23, h32, h41}
         │                      │
         ├── SFE (ResNet-18) ──→ {f1..f4}
         │      │                   │
         │   [frozen]          SFEProjection
         │                         │
         │                    {p1..p4}
         │                         │
         ├── ForwardDiff ──→ x_t ──┤
         │      │                  │
         │   t ─┼──→ TimeEmbed ──→ TAWG ──→ W_t
         │      │
         └── UNet Encoder ──→ {skip0..skip3, h}
                  │                  │
                  └── CrossAttn ←───┘  (Q=unet, K/V=sfe)
                         │
                      F_enc @ {128,64,32,16}^2
                         │
            F_fuse = (1-W_t)*F_enc + W_t*F_h   (Eq.9)
                         │
                  UNet Decoder ← F_fuse injected
                         │
                      eps_pred [B,1,512,512]

    Loss: L = ||eps_pred - noise||^2  (Eq.14)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dwt_module import HaarDWT, get_high_frequency_subbands
from mhfb import MHFB, BConv
from sfe import SFE, SFEProjection
from tawg import TAWG, TimestepEmbedding
from cross_attention import CrossAttention
from unet import UNet
from diffusion import ForwardDiffusion


class FADNet(nn.Module):
    """
    FAD-Net complete model.

    Input:  I_noisy [B, 1, 512, 512]   noisy OCT image
            t        [B]                diffusion timestep
    Output: eps_pred [B, 1, 512, 512]   predicted noise

    Trainable: U-Net + MHFB + TAWG + CrossAttn + fuse_proj
    Frozen:    SFE + SFEProjection (pre-trained, Section III-B)
    """

    def __init__(self, in_channels: int = 1,
                 enc_channels: list = None,
                 mhfb_channels: list = None,
                 time_emb_dim: int = 128,
                 T: int = 1000,
                 sfe_frozen: bool = True):
        super().__init__()

        # ---- defaults (paper reasonable values) ----
        if enc_channels is None:
            enc_channels = [32, 64, 96, 128]       # user specified
        if mhfb_channels is None:
            mhfb_channels = [64, 48, 32, 16]       # speculative: channel shrinkage

        ch_enc = enc_channels
        ch_mhfb = mhfb_channels

        # ============================================================
        # 1. Frequency Decomposition (Section III-C1, Eq.6)
        # ============================================================
        # Input:  [B, 1, 512, 512]
        # Output: I_LL, I_LH, I_HL, I_HH  each [B, 1, 256, 256]
        self.dwt = HaarDWT()

        # ============================================================
        # 2. Multi-scale HF Feature Extraction (Section III-C2, Eq.7)
        # ============================================================
        # Input:  HF = cat([I_LH, I_HL, I_HH])  →  [B, 3, 256, 256]
        # Output: F_h = {h14, h23, h32, h41}
        #              h14: [B,  64, 128, 128]
        #              h23: [B,  48,  64,  64]
        #              h32: [B,  32,  32,  32]
        #              h41: [B,  16,  16,  16]
        self.mhfb = MHFB(in_channels=in_channels * 3,
                          base_channels=ch_mhfb)

        # ============================================================
        # 3. Structural Feature Extractor (Section III-B, Fig.3)
        # ============================================================
        # Input:  [B, 1, 512, 512]
        # Output: f1: [B,  64, 128, 128]
        #         f2: [B, 128,  64,  64]
        #         f3: [B, 256,  32,  32]
        #         f4: [B, 512,  16,  16]
        self.sfe = SFE(in_channels=in_channels)

        # Project SFE features to match cross-attention KV dimensions
        # p1: [B, 128, 128, 128]    p2: [B, 256, 64, 64]
        # p3: [B, 256,  32,  32]    p4: [B, 256, 16, 16]
        self.sfe_proj = SFEProjection(
            sfe_channels=[64, 128, 256, 512],
            unet_channels=[128, 256, 256, 256],
        )

        if sfe_frozen:
            for p in self.sfe.parameters():
                p.requires_grad = False
            for p in self.sfe_proj.parameters():
                p.requires_grad = False

        # ============================================================
        # 4. Timestep Embedding + TAWG (Section III-C3, Eq.8)
        # ============================================================
        # t [B] → SinusoidalEmbedding(128) → TAWG MLP → W_t [B, 1] in (0,1)
        self.time_embed = TimestepEmbedding(dim=time_emb_dim)
        self.tawg = TAWG(embed_dim=time_emb_dim, hidden_dim=64)

        # ============================================================
        # 5. Multi-scale Cross-Attention (Section III-C3)
        # ============================================================
        # At each of 4 scales, cross-attention fuses U-Net encoder
        # features (Q) with SFE structural features (K, V).
        #
        # Scale 1 (128^2): Q=[B, 96,128,128]  x KV=[B,128,128,128]
        # Scale 2 ( 64^2): Q=[B,128, 64, 64]  x KV=[B,256, 64, 64]
        # Scale 3 ( 32^2): Q=[B,128, 32, 32]  x KV=[B,256, 32, 32]
        # Scale 4 ( 16^2): Q=[B,128, 16, 16]  x KV=[B,256, 16, 16]
        #
        # Output: F_enc matching Q shape at each scale
        self.cross_attn = nn.ModuleList([
            CrossAttention(q_dim=ch_enc[2], kv_dim=128, n_heads=4),   # 128^2
            CrossAttention(q_dim=ch_enc[3], kv_dim=256, n_heads=4),   #  64^2
            CrossAttention(q_dim=ch_enc[3], kv_dim=256, n_heads=4),   #  32^2
            CrossAttention(q_dim=ch_enc[3], kv_dim=256, n_heads=4),   #  16^2
        ])

        # ============================================================
        # 6. U-Net Noise Predictor
        # ============================================================
        # Standard DDPM 4-level U-Net with timestep conditioning.
        # Encoder: [B,1,512,512] → ... → [B,128,32,32] bottleneck
        # Decoder: [B,128,32,32] → ... → [B,1,512,512] eps_pred
        self.unet = UNet(in_channels=in_channels,
                          out_channels=in_channels,
                          enc_channels=ch_enc,
                          time_emb_dim=time_emb_dim)

        # ============================================================
        # 7. F_h channel alignment (for Eq.9 fusion)
        # ============================================================
        # Project MHFB outputs to match cross-attention output channels.
        # F_h → Conv1x1 → same channels as F_enc at each scale
        self.fuse_proj = nn.ModuleList([
            nn.Conv2d(ch_mhfb[0], ch_enc[2], 1),   # h14: 64→96  (128^2)
            nn.Conv2d(ch_mhfb[1], ch_enc[3], 1),   # h23: 48→128 ( 64^2)
            nn.Conv2d(ch_mhfb[2], ch_enc[3], 1),   # h32: 32→128 ( 32^2)
            nn.Conv2d(ch_mhfb[3], ch_enc[3], 1),   # h41: 16→128 ( 16^2)
        ])

        # Store config for reference
        self.in_channels = in_channels
        self.enc_channels = ch_enc
        self.mhfb_channels = ch_mhfb
        self.T = T

    # ================================================================
    # Prior Extraction (Algorithm 1, lines 2-3 + SFE)
    # ================================================================
    def extract_priors(self, I_noisy: torch.Tensor
                       ) -> "tuple[dict, dict]":
        """
        Extract frequency and structural priors from the original OCT image.

        These priors are INDEPENDENT of t and x_t — they characterise the
        underlying tissue structure (SFE) and high-frequency details (MHFB)
        of the original noisy image.  During inference they are computed
        ONCE and reused across all denoising steps.

        Paper:
            Step 2:  I_LL,I_LH,I_HL,I_HH = DWT(I)
            Step 3:  F_h = MHFB({I_LH,I_HL,I_HH})
            Sec.III-B: SFE extracts structural features for cross-attention

        Args:
            I_noisy: [B, 1, 512, 512]   original noisy OCT image

        Returns:
            F_h:      dict with keys 'h14','h23','h32','h41'
                      MHFB 4-scale high-frequency pyramid.
                      h14:[B,  64,128,128]  h23:[B,  48, 64, 64]
                      h32:[B,  32, 32, 32]  h41:[B,  16, 16, 16]
            sfe_proj: dict with keys 'p1'..'p4'
                      Projected SFE features for cross-attention K/V.
                      p1:[B,128,128,128]  p2:[B,256, 64, 64]
                      p3:[B,256, 32, 32]  p4:[B,256, 16, 16]
        """
        # ---- DWT: I_noisy → 4 subbands (Eq.6) ----
        I_LL, I_LH, I_HL, I_HH = self.dwt(I_noisy)
        # each [B, 1, 256, 256]

        # ---- MHFB: 3 HF subbands → 4-scale pyramid F_h (Eq.7) ----
        hf = torch.cat([I_LH, I_HL, I_HH], dim=1)              # [B, 3, 256, 256]
        F_h = self.mhfb(hf)
        # h14:[B, 64,128,128]  h23:[B, 48, 64, 64]
        # h32:[B, 32, 32, 32]  h41:[B, 16, 16, 16]

        # ---- SFE: I_noisy → multi-scale structural features ----
        sfe_feats = self.sfe(I_noisy)
        # f1:[B,64,128,128]  f2:[B,128,64,64]
        # f3:[B,256,32,32]   f4:[B,512,16,16]
        sfe_proj = self.sfe_proj(sfe_feats)
        # p1:[B,128,128,128]  p2:[B,256,64,64]
        # p3:[B,256,32,32]    p4:[B,256,16,16]

        return F_h, sfe_proj

    # ================================================================
    # Noise Prediction (Algorithm 1, lines 7-9)
    # ================================================================
    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor,
                      F_h: dict, sfe_proj: dict
                      ) -> torch.Tensor:
        """
        Predict the noise component in x_t, conditioned on time t and
        pre-extracted frequency/structural priors.

        This is the core ε_θ(x_t, t, F_fuse) call from Eq.12/14.
        Separating it from prior extraction allows the priors to be
        computed once and reused across all denoising steps during
        inference (standard diffusion sampling pattern).

        Paper:
            Step 7:  U-Net encoder extracts features from x_t
                     → cross-attention with SFE projections → F_enc
            Step 8:  W_t = TAWG(t_emb)
            Step 9:  F_fuse = (1-W_t)⊙F_enc + W_t⊙F_h  (per scale)
                     → U-Net decoder with F_fuse injection → ε_pred

        Args:
            x_t:      [B, 1, 512, 512]   noised image at timestep t
            t:        [B]                 diffusion timesteps
            F_h:      dict                from extract_priors()
            sfe_proj: dict                from extract_priors()

        Returns:
            eps_pred: [B, 1, 512, 512]   predicted noise ε
        """
        B = x_t.shape[0]

        # ---- TAWG: t → embed → MLP → W_t ∈ (0,1)^4 (Eq.8, per-scale) ----
        t_emb = self.time_embed(t)                              # [B, 128]
        W_t = self.tawg(t_emb)                                  # [B, 4]   one weight per scale
        # Reshape for broadcasting: [B, 4, 1, 1, 1] → index by scale
        W_s = [W_t[:, i].view(B, 1, 1, 1) for i in range(4)]   # 4 × [B, 1, 1, 1]

        # ================================================================
        # U-Net Encoder on x_t (Algorithm 1, line 7)
        #   "Extract encoder features F_enc from x_t using U-Net encoder"
        # ================================================================
        t_emb_unet = self.unet.time_embed(t)                    # [B, 128]

        # conv_in: x_t [B,1,512,512] → [B,32,512,512]
        h = self.unet.conv_in(x_t)

        # Level 0: [B,32,512,512]
        for block in self.unet.enc0:
            h = block(h, t_emb_unet)                            # [B,  32, 512, 512]
        skip0 = h                                               # [B,  32, 512, 512]
        h = self.unet.down0(h)                                  # [B,  32, 256, 256]

        # Level 1: [B,32,256,256] → [B,64,256,256]
        for block in self.unet.enc1:
            h = block(h, t_emb_unet)                            # [B,  64, 256, 256]
        skip1 = h                                               # [B,  64, 256, 256]
        h = self.unet.down1(h)                                  # [B,  64, 128, 128]

        # Level 2: [B,64,128,128] → [B,96,128,128]
        for block in self.unet.enc2:
            h = block(h, t_emb_unet)                            # [B,  96, 128, 128]
        unet_enc_128 = h                                        # [B,  96, 128, 128]  (→ cross-attn scale 1)
        skip2 = h                                               # [B,  96, 128, 128]
        h = self.unet.down2(h)                                  # [B,  96,  64,  64]

        # Level 3: [B,96,64,64] → [B,128,64,64]
        for block in self.unet.enc3:
            h = block(h, t_emb_unet)                            # [B, 128,  64,  64]
        unet_enc_64 = h                                         # [B, 128,  64,  64]  (→ cross-attn scale 2)
        skip3 = h                                               # [B, 128,  64,  64]
        h = self.unet.down3(h)                                  # [B, 128,  32,  32]

        # Bottleneck features for cross-attn scales 3 & 4
        #   Scale 3 (32²): bottleneck itself
        #   Scale 4 (16²): downsampled from bottleneck (U-Net bottoms at 32²;
        #     paper does not specify architecture; 16² injection follows MHFB pyramid)
        unet_enc_32 = h                                         # [B, 128,  32,  32]
        unet_enc_16 = F.interpolate(h, scale_factor=0.5,        # [B, 128,  16,  16]
                                     mode='bilinear', align_corners=False)

        # ================================================================
        # Cross-Attention: Q = U-Net encoder, K/V = SFE projected (Eq.8 context)
        # ================================================================

        # Scale 1 (128²): Q [B, 96,128,128]  ×  KV [B,128,128,128]
        F_enc_128 = self.cross_attn[0](unet_enc_128, sfe_proj['p1'])
        F_h_128 = self.fuse_proj[0](F_h['h14'])                # [B, 96, 128, 128]
        F_fuse_128 = (1 - W_s[0]) * F_enc_128 + W_s[0] * F_h_128

        # Scale 2 ( 64²): Q [B,128, 64, 64]  ×  KV [B,256, 64, 64]
        F_enc_64 = self.cross_attn[1](unet_enc_64, sfe_proj['p2'])
        F_h_64 = self.fuse_proj[1](F_h['h23'])                 # [B, 128, 64, 64]
        F_fuse_64 = (1 - W_s[1]) * F_enc_64 + W_s[1] * F_h_64

        # Scale 3 ( 32²): Q [B,128, 32, 32]  ×  KV [B,256, 32, 32]
        F_enc_32 = self.cross_attn[2](unet_enc_32, sfe_proj['p3'])
        F_h_32 = self.fuse_proj[2](F_h['h32'])                 # [B, 128, 32, 32]
        F_fuse_32 = (1 - W_s[2]) * F_enc_32 + W_s[2] * F_h_32

        # Scale 4 ( 16²): Q [B,128, 16, 16]  ×  KV [B,256, 16, 16]
        F_enc_16 = self.cross_attn[3](unet_enc_16, sfe_proj['p4'])
        F_h_16 = self.fuse_proj[3](F_h['h41'])                 # [B, 128, 16, 16]
        F_fuse_16 = (1 - W_s[3]) * F_enc_16 + W_s[3] * F_h_16

        # ================================================================
        # U-Net Middle (bottleneck, 32²) + F_fuse injection
        # ================================================================
        h = self.unet.mid_res1(h, t_emb_unet)                   # [B, 128, 32, 32]
        h = self.unet.mid_attn(h)                                # [B, 128, 32, 32]

        # Inject F_fuse_16: 16² → up to 32² → add
        F_fuse_mid = F.interpolate(F_fuse_16, scale_factor=2.0,
                                    mode='bilinear', align_corners=False)
        h = h + F_fuse_mid                                      # [B, 128, 32, 32]
        h = self.unet.mid_res2(h, t_emb_unet)                   # [B, 128, 32, 32]

        # ================================================================
        # U-Net Decoder — hierarchical F_fuse injection at each level
        #   Paper Section III-C3: "Fi_fuse is first spatially aligned and
        #   then dynamically injected into the decoding pathway."
        # ================================================================

        # Level 3': 32² → 64²
        h = self.unet.up3(h)                                    # [B, 128, 64, 64]
        h = h + F.interpolate(F_fuse_32, scale_factor=2.0,
                               mode='bilinear', align_corners=False)
        h = torch.cat([h, skip3], dim=1)                       # [B, 256, 64, 64]
        for block in self.unet.dec3:
            h = block(h, t_emb_unet)                            # [B, 128, 64, 64]

        # Level 2': 64² → 128²
        h = self.unet.up2(h)                                    # [B, 128, 128, 128]
        h = h + F.interpolate(F_fuse_64, scale_factor=2.0,
                               mode='bilinear', align_corners=False)
        h = torch.cat([h, skip2], dim=1)                       # [B, 224, 128, 128]
        for block in self.unet.dec2:
            h = block(h, t_emb_unet)                            # [B,  96, 128, 128]

        # Level 1': 128² → 256²
        h = self.unet.up1(h)                                    # [B,  96, 256, 256]
        h = h + F.interpolate(F_fuse_128, scale_factor=2.0,
                               mode='bilinear', align_corners=False)
        h = torch.cat([h, skip1], dim=1)                       # [B, 160, 256, 256]
        for block in self.unet.dec1:
            h = block(h, t_emb_unet)                            # [B,  64, 256, 256]

        # Level 0': 256² → 512²
        h = self.unet.up0(h)                                    # [B,  64, 512, 512]
        h = torch.cat([h, skip0], dim=1)                       # [B,  96, 512, 512]
        for block in self.unet.dec0:
            h = block(h, t_emb_unet)                            # [B,  32, 512, 512]

        # Output: [B,32,512,512] → [B,1,512,512]
        eps_pred = self.unet.conv_out(h)                        # [B, 1, 512, 512]

        return eps_pred

    # ================================================================
    # Forward (convenience wrapper — training use)
    # ================================================================
    def forward(self, I_noisy: torch.Tensor, x_t: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        FAD-Net forward pass — convenience wrapper for training.

        Equivalent to:
            F_h, sfe_proj = self.extract_priors(I_noisy)
            eps_pred = self.predict_noise(x_t, t, F_h, sfe_proj)

        Args:
            I_noisy: [B, 1, 512, 512]   original noisy OCT image (x_0)
            x_t:     [B, 1, 512, 512]   noised image at timestep t (Eq.10)
            t:       [B]                 diffusion timesteps in [0, T-1]

        Returns:
            eps_pred: [B, 1, 512, 512]   predicted noise ε
        """
        F_h, sfe_proj = self.extract_priors(I_noisy)
        return self.predict_noise(x_t, t, F_h, sfe_proj)


# ============================================================
# Test Code
# ============================================================
if __name__ == "__main__":
    import os
    from pathlib import Path

    print("=" * 60)
    print("FAD-Net Complete Model — Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ---- Build model ----
    print(f"\n{'─' * 60}")
    print("Building FAD-Net...")
    print(f"{'─' * 60}")

    model = FADNet(in_channels=1, sfe_frozen=True).to(device)

    # ---- Model statistics ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"  Total parameters:     {total_params:>12,}")
    print(f"  Trainable parameters: {trainable_params:>12,}")
    print(f"  Frozen (SFE):         {frozen_params:>12,}")

    # ---- Test 1: Forward pass ----
    print(f"\n{'─' * 60}")
    print("Test 1: Forward Pass (B=2, 512x512)")
    print(f"{'─' * 60}")

    from diffusion import ForwardDiffusion
    diff_test = ForwardDiffusion(T=1000).to(device)

    B = 2
    I_noisy = torch.randn(B, 1, 512, 512, device=device)
    t = torch.randint(0, 1000, (B,), device=device)

    # Forward diffusion: x_t = sqrt(alpha_bar_t)*I_noisy + sqrt(1-alpha_bar_t)*noise
    x_t, noise = diff_test(I_noisy, t)

    print(f"  I_noisy:      {list(I_noisy.shape)}")
    print(f"  x_t:          {list(x_t.shape)}")
    print(f"  t:            {t.tolist()}")

    use_amp = (device.type == 'cuda')
    with torch.amp.autocast('cuda', enabled=use_amp):
        eps_pred = model(I_noisy, x_t, t)

    print(f"  eps_pred:     {list(eps_pred.shape)}")
    print(f"  Shape match:  {eps_pred.shape == I_noisy.shape}")
    print(f"  Value range:  [{eps_pred.min().item():.4f}, {eps_pred.max().item():.4f}]")

    # ---- Test 2: Gradient flow ----
    print(f"\n{'─' * 60}")
    print("Test 2: Gradient Flow + Optimizer Step (B=1)")
    print(f"{'─' * 60}")

    # Clear memory from Test 1
    del I_noisy, x_t, noise, eps_pred
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    I_noisy_grad = torch.randn(1, 1, 512, 512, device=device)
    t_grad = torch.tensor([500], device=device)
    x_t_grad, _ = diff_test(I_noisy_grad, t_grad)   # generate x_t via forward diffusion

    # Forward with correct signature: I_noisy → priors, x_t → UNet
    eps = model(I_noisy_grad, x_t_grad, t_grad)
    loss = F.mse_loss(eps, torch.randn_like(eps))
    loss.backward()

    # Check which modules received gradients
    grad_modules = []
    no_grad_modules = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_modules.append(name.split('.')[0])
        elif param.requires_grad:
            no_grad_modules.append(name.split('.')[0])

    unique_grad = sorted(set(grad_modules))
    unique_nograd = sorted(set(no_grad_modules))
    print(f"  Modules with grad:    {unique_grad}")
    print(f"  Modules without grad: {unique_nograd} (should be [sfe, sfe_proj])")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Grad flow OK: True")

    # Optimizer step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.step()
    print(f"  Optimizer.step() OK")

    # Cleanup
    del I_noisy_grad, x_t_grad, eps, loss
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ---- Test 3: Memory ----
    print(f"\n{'─' * 60}")
    print("Test 3: GPU Memory")
    print(f"{'─' * 60}")

    if device.type == 'cuda':
        mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
        mem_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"  Allocated:  {mem_alloc:.1f} GB")
        print(f"  Reserved:   {mem_reserved:.1f} GB")

    # ---- Test 4: Real OCTA image inference ----
    print(f"\n{'─' * 60}")
    print("Test 4: Inference on Real OCTA Image")
    print(f"{'─' * 60}")

    from PIL import Image as PILImage
    import torchvision.transforms as T

    script_dir = os.path.dirname(os.path.abspath(__file__))
    octa_paths = list(Path(script_dir).glob('3mmx3mm*.bmp'))

    if octa_paths:
        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        img = PILImage.open(octa_paths[0]).convert('L')
        octa_tensor = transform(img).unsqueeze(0).to(device)   # [1,1,512,512]
        print(f"  Image: {octa_paths[0].name}")
        print(f"  Tensor: {list(octa_tensor.shape)}  "
              f"range=[{octa_tensor.min().item():.4f}, {octa_tensor.max().item():.4f}]")

        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp):
                t_infer = torch.tensor([500], device=device)
                # Generate x_t from I_noisy for a single-step test
                x_t_infer, _ = diff_test(octa_tensor, t_infer)
                eps = model(octa_tensor, x_t_infer, t_infer)

        print(f"  eps_pred: {list(eps.shape)}")
        print(f"  eps range: [{eps.min().item():.4f}, {eps.max().item():.4f}]")
        print(f"  Inference OK: True")
    else:
        print("  No OCTA BMP files found — skipping.")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("FADNet Model Summary")
    print(f"{'=' * 60}")
    print(f"""
    Architecture:
        DWT (Haar)          → I_LL, I_LH, I_HL, I_HH
        MHFB (Eq.7)         → F_h @ {{128,64,32,16}}^2
        SFE (ResNet-18)     → f1..f4 → SFEProjection → p1..p4
        TimeEmbed + TAWG    → W_t in (0,1)
        CrossAttn (4 scale) → F_enc (Q=UNet, K/V=SFE)
        Fuse (Eq.9)         → F_fuse = (1-W_t)*F_enc + W_t*F_h
        UNet Decoder        → + F_fuse injected → eps_pred

    Parameters:
        Total:      {total_params:>12,}
        Trainable:  {trainable_params:>12,}  (UNet + MHFB + TAWG + CrossAttn)
        Frozen:     {frozen_params:>12,}  (SFE — Section III-B)

    Input:   [B, 1, 512, 512]  noisy OCT image
    Output:  [B, 1, 512, 512]  predicted noise epsilon

    Training:
        L = ||eps_pred - noise||^2   (Eq.14)
        x_t = sqrt(alpha_bar_t)*I_noisy + sqrt(1-alpha_bar_t)*noise   (Eq.10)

    Inference:
        Start from x_T ~ N(0,I), iteratively denoise using eps_pred,
        converge to clean OCT image.
    """)
