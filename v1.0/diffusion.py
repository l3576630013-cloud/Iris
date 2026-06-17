"""
FAD-Net DDPM Diffusion Process (Forward + Reverse)
====================================================
Paper: FAD-Net, IEEE TIM 2026, Section III-D, Eq.(10)-(13)

Forward  (Eq.10): x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
Reverse (Eq.11):  p_theta(x_{t-1}|x_t) = N(x_{t-1}; mu_theta, sigma_t^2 I)
         (Eq.12):  mu_theta = 1/sqrt(alpha_t)*(x_t - (1-alpha_t)/sqrt(1-alpha_bar_t)*eps_theta)
         (Eq.13):  x_hat_0 = 1/sqrt(alpha_bar_t)*(x_t - sqrt(1-alpha_bar_t)*eps_theta)

Schedule:
    T = 1000
    beta_t: linear from 1e-4 to 0.02  (DDPM standard)
    alpha_t = 1 - beta_t
    alpha_bar_t = cumprod(alpha_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image as PILImage


class ForwardDiffusion(nn.Module):
    """
    DDPM forward diffusion: q(x_t | x_0).

    Precomputes noise schedule and implements Eq.(10).
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        super().__init__()
        self.T = T

        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)

    def forward(self, x_0: torch.Tensor, t: torch.Tensor,
                noise: torch.Tensor = None):
        """
        Eq.(10): x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps

        Args:
            x_0:   [B, C, H, W]   input (noisy OCT in FAD-Net)
            t:     [B]             timestep indices in [0, T-1]
            noise: [B, C, H, W]   optional; samples N(0,I) if None

        Returns:
            x_t:   [B, C, H, W]
            noise: [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        x_t = sqrt_ac * x_0 + sqrt_omac * noise
        return x_t, noise

    def get_coeff(self, t: torch.Tensor) -> dict:
        """Return alpha-related coefficients at timestep t."""
        return {
            'alpha': self.alphas[t],
            'alpha_cumprod': self.alphas_cumprod[t],
            'beta': self.betas[t],
            'sqrt_alpha_cumprod': self.sqrt_alphas_cumprod[t],
            'sqrt_one_minus_alpha_cumprod': self.sqrt_one_minus_alphas_cumprod[t],
        }


class ReverseDiffusion(nn.Module):
    """
    DDPM reverse diffusion: p(x_{t-1} | x_t).

    Implements the iterative denoising sampling loop.
    Also provides DDIM accelerated sampling.
    """

    def __init__(self, fd: ForwardDiffusion):
        super().__init__()
        self.fd = fd
        self.T = fd.T

        sqrt_recip_alphas = torch.sqrt(1.0 / fd.alphas)
        coef_eps = (1.0 - fd.alphas) / fd.sqrt_one_minus_alphas_cumprod
        posterior_variance = fd.betas.clone()
        posterior_variance[0] = 0.0

        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('coef_eps', coef_eps)
        self.register_buffer('posterior_variance', posterior_variance)

    def reverse_step(self, x_t: torch.Tensor, t: torch.Tensor,
                     eps_pred: torch.Tensor,
                     add_noise: bool = True) -> torch.Tensor:
        """
        Single DDPM reverse step: x_t -> x_{t-1}.

        Args:
            x_t:       [B, C, H, W]
            t:         [B]
            eps_pred:  [B, C, H, W]
            add_noise: False at t=0

        Returns:
            x_{t-1}: [B, C, H, W]
        """
        sr = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        ce = self.coef_eps[t].view(-1, 1, 1, 1)
        st = torch.sqrt(self.posterior_variance[t]).view(-1, 1, 1, 1)

        mu_theta = sr * (x_t - ce * eps_pred)

        if add_noise:
            return mu_theta + st * torch.randn_like(x_t)
        return mu_theta

    @torch.no_grad()
    def sample(self, noise_pred_fn, shape: tuple,
               verbose: bool = True) -> torch.Tensor:
        """
        Full DDPM sampling: x_T ~ N(0,I) -> x_0.

        Args:
            noise_pred_fn: callable(x_t, t) -> eps_pred
            shape:         (B, C, H, W)

        Returns:
            x_0: [B, C, H, W]
        """
        device = self.fd.betas.device
        B = shape[0]
        x_t = torch.randn(shape, device=device)

        for t_idx in reversed(range(0, self.T)):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)
            eps_pred = noise_pred_fn(x_t, t)
            x_t = self.reverse_step(x_t, t, eps_pred, add_noise=(t_idx > 0))

            if verbose and t_idx % 100 == 0:
                print(f"    t={t_idx:4d}  mean={x_t.mean().item():.4f}  "
                      f"std={x_t.std().item():.4f}")

        return x_t

    @torch.no_grad()
    def sample_ddim(self, noise_pred_fn, shape: tuple,
                    ddim_steps: int = 50, eta: float = 0.0,
                    verbose: bool = True) -> torch.Tensor:
        """
        DDIM accelerated sampling (Song et al., ICLR 2021).

        Args:
            noise_pred_fn: callable(x_t, t) -> eps_pred
            shape:         (B, C, H, W)
            ddim_steps:    number of steps (paper Fig.12: 30-50)
            eta:           0=DDIM, 1=DDPM

        Returns:
            x_0: [B, C, H, W]
        """
        device = self.fd.betas.device
        B = shape[0]

        step_indices = torch.linspace(self.T - 1, 0, ddim_steps,
                                       dtype=torch.long, device=device)
        x_t = torch.randn(shape, device=device)

        for i in range(ddim_steps):
            t_curr = step_indices[i].item()
            t_next = step_indices[i + 1].item() if i < ddim_steps - 1 else -1

            t_batch = torch.full((B,), t_curr, device=device, dtype=torch.long)
            eps_pred = noise_pred_fn(x_t, t_batch)

            ac = self.fd.alphas_cumprod[t_curr]
            s_ac = self.fd.sqrt_alphas_cumprod[t_curr]
            s_omac = self.fd.sqrt_one_minus_alphas_cumprod[t_curr]

            x0_pred = (x_t - s_omac * eps_pred) / s_ac
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            if t_next < 0:
                x_t = x0_pred
                break

            ac_next = self.fd.alphas_cumprod[t_next]
            dir_xt = torch.sqrt(1.0 - ac_next) * eps_pred

            if eta > 0:
                sigma = eta * torch.sqrt(
                    (1.0 - ac_next) / (1.0 - ac) * (1.0 - ac / ac_next)
                )
                dir_xt = torch.sqrt(1.0 - ac_next - sigma ** 2) * eps_pred
                x_t = (torch.sqrt(ac_next) * x0_pred + dir_xt +
                       sigma * torch.randn_like(x_t))
            else:
                x_t = torch.sqrt(ac_next) * x0_pred + dir_xt

            if verbose and (i % 10 == 0 or i == ddim_steps - 1):
                print(f"    DDIM {i:3d}/{ddim_steps}  t={t_curr:4d}  "
                      f"mean={x_t.mean().item():.4f}")

        return x_t


# ============================================================
# Test Code
# ============================================================
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("FAD-Net DDPM Diffusion -- Full Test (Forward + Reverse)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    save_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- Instantiate ----
    fd = ForwardDiffusion(T=1000, beta_start=1e-4, beta_end=0.02)
    print(f"\nT = {fd.T}")
    print(f"beta_1 = {fd.betas[0].item():.6f}, beta_T = {fd.betas[-1].item():.6f}")
    print(f"alpha_bar_0 = {fd.alphas_cumprod[0].item():.6f}")
    print(f"alpha_bar_999 = {fd.alphas_cumprod[-1].item():.8f}")

    # ================================================================
    # Test 1: Noise schedule
    # ================================================================
    print(f"\n{'─' * 60}")
    print("Test 1: Noise Schedule")
    print(f"{'─' * 60}")

    betas_np = fd.betas.cpu().numpy()
    alphas_cumprod_np = fd.alphas_cumprod.cpu().numpy()
    snr = alphas_cumprod_np / (1 - alphas_cumprod_np)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(betas_np)
    axes[0].set_title('beta_t (linear)')
    axes[1].plot(alphas_cumprod_np)
    axes[1].set_title('alpha_bar_t')
    axes[2].semilogy(snr)
    axes[2].set_title('SNR (log scale)')
    plt.tight_layout()
    schedule_path = os.path.join(save_dir, 'noise_schedule.png')
    fig.savefig(schedule_path, dpi=100)
    plt.close(fig)
    print(f"  Saved: {schedule_path}")

    for t_idx in [0, 50, 100, 250, 500, 750, 999]:
        print(f"  t={t_idx:4d}  beta={fd.betas[t_idx].item():.6f}  "
              f"alpha_bar={fd.alphas_cumprod[t_idx].item():.6f}  "
              f"sqrt(1-ac)={fd.sqrt_one_minus_alphas_cumprod[t_idx].item():.6f}")

    # ================================================================
    # Test 2: Forward diffusion visualization
    # ================================================================
    print(f"\n{'─' * 60}")
    print("Test 2: Forward Diffusion Visualization")
    print(f"{'─' * 60}")

    y, x = torch.meshgrid(torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256), indexing='ij')
    test_img = 0.3 + 0.2 * torch.sin(x * 10) * torch.cos(y * 8)
    test_img += 0.1 * torch.randn(256, 256)
    test_img = test_img.clamp(0, 1).float()
    test_img = test_img.unsqueeze(0).unsqueeze(0).to(device)
    print(f"  x_0 shape: {list(test_img.shape)}")

    fd = fd.to(device)
    t_steps = [0, 50, 100, 250, 500, 750, 999]
    fig, axes = plt.subplots(1, len(t_steps), figsize=(16, 2.5))

    with torch.no_grad():
        for idx, t_val in enumerate(t_steps):
            t_tensor = torch.tensor([t_val], device=device)
            x_t, _ = fd(test_img, t_tensor)
            axes[idx].imshow(x_t[0, 0].cpu().numpy(), cmap='gray', vmin=-1, vmax=2)
            axes[idx].set_title(f't={t_val}')
            axes[idx].axis('off')
    plt.tight_layout()
    diffuse_path = os.path.join(save_dir, 'forward_diffusion.png')
    fig.savefig(diffuse_path, dpi=100)
    plt.close(fig)
    print(f"  Saved: {diffuse_path}")

    # ================================================================
    # Test 3: Random timestep batch
    # ================================================================
    print(f"\n{'─' * 60}")
    print("Test 3: Random Timestep Batch (training simulation)")
    print(f"{'─' * 60}")

    B = 4
    x_0_batch = torch.randn(B, 1, 512, 512, device=device)
    t_batch = torch.randint(0, 1000, (B,), device=device)
    print(f"  x_0: {list(x_0_batch.shape)}  |  t: {t_batch.tolist()}")

    with torch.no_grad():
        x_t_batch, noise_batch = fd(x_0_batch, t_batch)
    print(f"  x_t: {list(x_t_batch.shape)}  |  noise: {list(noise_batch.shape)}")
    for b in range(B):
        diff = (x_t_batch[b] - x_0_batch[b]).abs().mean().item()
        print(f"    b={b} t={t_batch[b].item():3d}  |x_t-x_0|={diff:.6f}")

    # ================================================================
    # Test 4: Boundary cases
    # ================================================================
    print(f"\n{'─' * 60}")
    print("Test 4: Boundary Cases (t=0, t=999)")
    print(f"{'─' * 60}")

    x_test = torch.ones(1, 1, 64, 64, device=device) * 0.5
    with torch.no_grad():
        xt0, _ = fd(x_test, torch.tensor([0], device=device))
        xt999, _ = fd(x_test, torch.tensor([999], device=device))
    print(f"  t=0:   |x_t-x_0|_max = {(xt0-x_test).abs().max().item():.6f}")
    print(f"  t=999: mean={xt999.mean().item():.4f}, std={xt999.std().item():.4f}")

    # ================================================================
    # Test 5: Determinism
    # ================================================================
    print(f"\n{'─' * 60}")
    print("Test 5: Determinism (fixed noise)")
    print(f"{'─' * 60}")

    torch.manual_seed(42)
    x_fixed = torch.randn(1, 1, 32, 32, device=device)
    noise_fixed = torch.randn(1, 1, 32, 32, device=device)
    with torch.no_grad():
        xt_a, _ = fd(x_fixed, torch.tensor([500], device=device), noise=noise_fixed)
        xt_b, _ = fd(x_fixed, torch.tensor([500], device=device), noise=noise_fixed)
    print(f"  Same inputs -> same x_t: {torch.allclose(xt_a, xt_b)}")

    # ================================================================
    # Test 6: Single reverse step
    # ================================================================
    print(f"\n{'─' * 60}")
    print("Test 6: Single Reverse Step (Eq.11-12)")
    print(f"{'─' * 60}")

    rd = ReverseDiffusion(fd).to(device)

    x_t_test = torch.randn(2, 1, 64, 64, device=device)
    t_test = torch.tensor([500, 500], device=device)
    eps_fake = torch.randn_like(x_t_test) * 0.1

    with torch.no_grad():
        x_prev = rd.reverse_step(x_t_test, t_test, eps_fake, add_noise=True)
        x_prev_det = rd.reverse_step(x_t_test, torch.tensor([0, 0], device=device),
                                      eps_fake, add_noise=False)
        x_prev_noisy = rd.reverse_step(x_t_test, torch.tensor([0, 0], device=device),
                                        eps_fake, add_noise=True)

    print(f"  x_t: {list(x_t_test.shape)}  ->  x_{{t-1}}: {list(x_prev.shape)}")
    print(f"  x_prev != x_t: {not torch.allclose(x_prev, x_t_test, atol=1e-4)}")
    diff_t0 = (x_prev_det - x_prev_noisy).abs().max().item()
    print(f"  t=0 det==noisy max diff: {diff_t0:.10f} (should be ~0)")

    # ================================================================
    # Test 7: Full sampling loop (mock UNet)
    # ================================================================
    print(f"\n{'─' * 60}")
    print("Test 7: Full DDPM Sampling (1000 steps, 64x64)")
    print(f"{'─' * 60}")

    class MockUNet(nn.Module):
        def forward(self, x_t, t):
            return torch.randn_like(x_t) * 0.1

    mock_unet = MockUNet().to(device)
    shape = (1, 1, 64, 64)

    print(f"  Sampling from pure noise (64x64, 1000 steps)...")
    with torch.no_grad():
        x0_sample = rd.sample(mock_unet, shape, verbose=False)
    print(f"  x_0: {list(x0_sample.shape)}  "
          f"range=[{x0_sample.min().item():.4f}, {x0_sample.max().item():.4f}]")

    # ================================================================
    # Test 8: DDIM sampling
    # ================================================================
    print(f"\n{'─' * 60}")
    print("Test 8: DDIM Sampling (50 steps)")
    print(f"{'─' * 60}")

    with torch.no_grad():
        x0_ddim = rd.sample_ddim(mock_unet, shape, ddim_steps=50, eta=0.0, verbose=True)
    print(f"  DDIM x_0: {list(x0_ddim.shape)}  "
          f"range=[{x0_ddim.min().item():.4f}, {x0_ddim.max().item():.4f}]")

    # ================================================================
    # Test 9: Reverse visualization
    # ================================================================
    print(f"\n{'─' * 60}")
    print("Test 9: Reverse Sampling Visualization")
    print(f"{'─' * 60}")

    shape_viz = (1, 1, 128, 128)
    snapshots = {}
    x_t = torch.randn(shape_viz, device=device)
    snapshots[999] = x_t.clone()

    with torch.no_grad():
        for t_idx in reversed(range(0, 1000)):
            t = torch.tensor([t_idx], device=device)
            eps_pred = mock_unet(x_t, t)
            x_t = rd.reverse_step(x_t, t, eps_pred, add_noise=(t_idx > 0))
            if t_idx in [750, 500, 250, 100, 50, 0]:
                snapshots[t_idx] = x_t.clone()

    fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 2.5))
    for idx, (t_val, img) in enumerate(sorted(snapshots.items(), reverse=True)):
        axes[idx].imshow(img[0, 0].cpu().numpy(), cmap='gray')
        axes[idx].set_title(f't={t_val}', fontsize=10)
        axes[idx].axis('off')
    plt.tight_layout()
    reverse_path = os.path.join(save_dir, 'reverse_diffusion.png')
    fig.savefig(reverse_path, dpi=100)
    plt.close(fig)
    print(f"  Saved: {reverse_path}")
    print(f"  Snapshots: t = {sorted(snapshots.keys(), reverse=True)}")

    # ================================================================
    # Summary
    # ================================================================
    a0 = fd.alphas_cumprod[0].item()
    aT = fd.alphas_cumprod[-1].item()
    print(f"\n{'=' * 60}")
    print("DDPM Diffusion Module Summary")
    print(f"{'=' * 60}")
    print(f"""
    Forward (Eq.10):
        x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon

    Reverse (Eq.11-12):
        mu_theta = 1/sqrt(alpha_t)*(x_t - (1-alpha_t)/sqrt(1-alpha_bar_t)*eps_theta)
        x_{{t-1}} = mu_theta + sigma_t * z   (t > 0)

    Schedule:
        T = 1000, beta: linear [1e-4, 0.02]
        alpha_bar_0   = {a0:.6f}
        alpha_bar_999 = {aT:.8f}

    Classes:
        ForwardDiffusion     - q(x_t | x_0)
        ReverseDiffusion     - p(x_{{t-1}} | x_t)
          .reverse_step()    - single denoising step
          .sample()          - full 1000-step DDPM loop
          .sample_ddim()     - DDIM accelerated (50 steps, per Fig.12)

    Files saved:
        {schedule_path}
        {diffuse_path}
        {reverse_path}
    """)
