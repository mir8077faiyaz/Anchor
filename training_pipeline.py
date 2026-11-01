"""
training_pipeline.py


pu_dataset / data_gen / info remain black boxes; this file only assumes a
DataLoader yields (clean_cube, label) batches, clean_cube: (B, C, H, W).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from proto_etf_modules import ProtoEncoder, generate_etf_simplex


# --------------------------------------------------------------------------
# Training-time perturbations (generic / simulated -- NOT HyperTTA)
# --------------------------------------------------------------------------
def spectral_perturb(cube: torch.Tensor, severity: float) -> torch.Tensor:
    """cube: (B, C, H, W). One of: band-wise gaussian noise, band dropout,
    spectral smoothing -- chosen randomly each call."""
    B, C, H, W = cube.shape
    choice = torch.randint(0, 3, (1,)).item()

    if choice == 0:
        # band-wise gaussian noise, independent sigma per band
        sigma = torch.rand(1, C, 1, 1, device=cube.device) * severity
        noise = torch.randn_like(cube) * sigma
        return cube + noise

    elif choice == 1:
        # band dropout: zero out a random subset of bands
        drop_frac = 0.1 + 0.4 * severity  # up to 50% of bands
        mask = (torch.rand(1, C, 1, 1, device=cube.device) > drop_frac).float()
        return cube * mask

    else:
        # spectral smoothing: small conv along band axis
        k = 3
        pad = k // 2
        weight = torch.ones(1, 1, k, device=cube.device) / k
        weight = weight * severity + torch.tensor(
            [0.0, 1.0, 0.0], device=cube.device
        ).view(1, 1, k) * (1 - severity)
        x = cube.permute(0, 2, 3, 1).reshape(-1, 1, C)  # (B*H*W, 1, C)
        x = F.pad(x, (pad, pad), mode="replicate")
        x = F.conv1d(x, weight)
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


def spatial_perturb(cube: torch.Tensor, severity: float) -> torch.Tensor:
    """One of: gaussian blur, spatial gaussian noise, block occlusion."""
    B, C, H, W = cube.shape
    choice = torch.randint(0, 3, (1,)).item()

    if choice == 0:
        # gaussian blur via depthwise conv, kernel size grows with severity
        k = 3 if severity < 0.5 else 5
        sigma = 0.5 + 1.5 * severity
        coords = torch.arange(k, dtype=torch.float32, device=cube.device) - k // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).view(1, 1, k, 1)
        g2 = g.transpose(2, 3)
        weight_h = g.repeat(C, 1, 1, 1)
        weight_v = g2.repeat(C, 1, 1, 1)
        x = F.conv2d(cube, weight_h, padding=(k // 2, 0), groups=C)
        x = F.conv2d(x, weight_v, padding=(0, k // 2), groups=C)
        return x

    elif choice == 1:
        sigma = severity * cube.std()
        return cube + torch.randn_like(cube) * sigma

    else:
        # block occlusion: zero a random contiguous patch
        bh = max(1, int(H * (0.1 + 0.3 * severity)))
        bw = max(1, int(W * (0.1 + 0.3 * severity)))
        out = cube.clone()
        for b in range(B):
            r0 = torch.randint(0, max(1, H - bh + 1), (1,)).item()
            c0 = torch.randint(0, max(1, W - bw + 1), (1,)).item()
            out[b, :, r0:r0 + bh, c0:c0 + bw] = 0.0
        return out


def shot_noise_perturb(cube: torch.Tensor, severity: float) -> torch.Tensor:
    """Two shot-noise-like regimes, chosen randomly:
      0: signal-dependent (heteroscedastic) Gaussian noise -- sigma scales
         with sqrt(|signal|), same qualitative behavior as physical shot
         noise (brighter signal -> more absolute noise).
      1: severe low-photon-count / quantization regime -- values are pushed
         toward near-zero and quantized into very few discrete levels.
         This is closer to what the real HyperTTA Poisson corruption
         actually looks like in practice (observed min=0, max=5, mean=0.075
         on raw scale -- a near-binary/quantized signal, not smooth noise),
         which a pure Gaussian model can't represent no matter the severity."""
    choice = torch.randint(0, 2, (1,)).item()

    if choice == 0:
        sigma = severity * torch.sqrt(torch.clamp(cube.abs(), min=1e-3))
        return cube + torch.randn_like(cube) * sigma

    else:
        B, C, H, W = cube.shape
        scale_down = 0.05 + 0.25 * (1.0 - severity)  # higher severity -> pushed closer to zero
        dim_cube = cube * scale_down

        num_levels = max(2, int(round(10 - 8 * severity)))  # higher severity -> fewer levels
        c_min = dim_cube.amin(dim=(1, 2, 3), keepdim=True)
        c_max = dim_cube.amax(dim=(1, 2, 3), keepdim=True)
        span = (c_max - c_min).clamp(min=1e-6)

        normalized01 = (dim_cube - c_min) / span
        quantized01 = torch.round(normalized01 * (num_levels - 1)) / (num_levels - 1)
        return quantized01 * span + c_min


# def apply_training_perturbations(
#     cube: torch.Tensor,
#     p_spectral: float = 0.5,
#     p_spatial: float = 0.5,
#     p_shot: float = 0.5,
# ) -> torch.Tensor:
#     """Applies each perturbation family independently with its own
#     activation probability and randomly sampled severity in [0.2, 1.0]."""
#     out = cube
#     if torch.rand(1).item() < p_spectral:
#         out = spectral_perturb(out, severity=float(torch.empty(1).uniform_(0.2, 1.0)))
#     if torch.rand(1).item() < p_spatial:
#         out = spatial_perturb(out, severity=float(torch.empty(1).uniform_(0.2, 1.0)))
#     if torch.rand(1).item() < p_shot:
#         out = shot_noise_perturb(out, severity=float(torch.empty(1).uniform_(0.2, 1.0)))
#     return out

def apply_training_perturbations(
    cube: torch.Tensor,
    p_spectral: float = 0.5,
    p_spatial: float = 0.5,
    p_shot: float = 0.5,
) -> torch.Tensor:
    """
    Applies perturbation families independently PER SAMPLE.

    Each sample independently decides whether to receive:
      - spectral perturbation
      - spatial perturbation
      - shot perturbation

    Each activated family gets its own independently sampled severity
    in [0.2, 1.0].
    """

    B = cube.shape[0]
    outputs = []

    for b in range(B):
        out = cube[b:b+1]

        # Spectral
        if torch.rand(1, device=cube.device).item() < p_spectral:
            severity = float(
                torch.empty(1, device=cube.device).uniform_(0.2, 1.0)
            )
            out = spectral_perturb(out, severity=severity)

        # Spatial
        if torch.rand(1, device=cube.device).item() < p_spatial:
            severity = float(
                torch.empty(1, device=cube.device).uniform_(0.2, 1.0)
            )
            out = spatial_perturb(out, severity=severity)

        # Shot
        if torch.rand(1, device=cube.device).item() < p_shot:
            severity = float(
                torch.empty(1, device=cube.device).uniform_(0.2, 1.0)
            )
            out = shot_noise_perturb(out, severity=severity)

        outputs.append(out)

    return torch.cat(outputs, dim=0)

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int, patch_size: int, base_hw: int = 4):
        super().__init__()
        self.base_hw = base_hw
        self.patch_size = patch_size
        base_dim = 128
        self.fc = nn.Linear(latent_dim, base_dim * base_hw * base_hw)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(base_dim, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.GELU(),
        )
        self.out_conv = nn.Conv2d(32, out_channels, 3, padding=1)

    def forward(self, latent):
        B = latent.shape[0]
        x = self.fc(latent).view(B, 128, self.base_hw, self.base_hw)
        x = self.up(x)
        x = F.interpolate(x, size=(self.patch_size, self.patch_size), mode='bilinear', align_corners=False)
        return self.out_conv(x)


def sad_loss(pred, target, eps=1e-8):
    p = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
    t = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])
    cos = F.cosine_similarity(p, t, dim=-1, eps=eps).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos).mean()

def cross_degradation_supcon_loss(clean_embeddings, noisy_embeddings, labels, temperature=0.1):
    """Symmetric same-class clean/degraded SupCon; BOTH views receive gradients."""
    clean = F.normalize(clean_embeddings, dim=-1)
    noisy = F.normalize(noisy_embeddings, dim=-1)
    pos = labels[:, None].eq(labels[None, :])

    def direction(a, b):
        logits = (a @ b.t()) / temperature
        logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        count = pos.sum(1).clamp_min(1)
        return -((logp * pos.float()).sum(1) / count).mean()

    return 0.5 * (direction(noisy, clean) + direction(clean, noisy))


def per_sample_ce_loss(embeddings, labels, etf_vertices, temperature=0.1):
    logits = (F.normalize(embeddings, dim=-1) @ etf_vertices.t()) / temperature
    return F.cross_entropy(logits, labels)


def _rbf_kernel(x, eps=1e-6):
    x = F.normalize(x, dim=-1)
    dist2 = torch.cdist(x, x).pow(2)
    off = dist2[~torch.eye(x.size(0), dtype=torch.bool, device=x.device)]
    sigma2 = off.detach().median().clamp_min(eps)
    return torch.exp(-dist2 / (2.0 * sigma2))



def rd_loss(x, y):
    """Normalized kernel-alignment dependence penalty (RD loss).

    This is a stable mini-batch implementation of the TIP paper's principle:
    invariant and private representations should be statistically independent.
    """
    if x.size(0) < 3:
        return x.sum() * 0.0
    K = _rbf_kernel(x)
    L = _rbf_kernel(y)
    n = K.size(0)
    H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    num = (Kc * Lc).sum()
    den = torch.sqrt((Kc * Kc).sum().clamp_min(1e-12) * (Lc * Lc).sum().clamp_min(1e-12))
    return num / den

def training_step(
    model, optimizer, etf_vertices,
    clean_cube, noisy_cube, labels,
    lambda_ce=1.0, ce_temperature=0.1,
    lambda_cd=0.10, cd_temperature=0.1, lambda_rd=0.05,
):
    optimizer.zero_grad(set_to_none=True)

    clean_f = model.forward_features(clean_cube)
    noisy_f = model.forward_features(noisy_cube)
    zc = clean_f['invariant']
    zn = noisy_f['invariant']

    loss_ce = per_sample_ce_loss(zn, labels, etf_vertices, ce_temperature)
    loss_cd = cross_degradation_supcon_loss(zc, zn, labels, cd_temperature)
    loss_rd = rd_loss(
        noisy_f['invariant'],
        noisy_f['private']
    )

    loss = (
        lambda_ce * loss_ce
        + lambda_cd * loss_cd
        + lambda_rd * loss_rd
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    return {
        'loss_total': loss.item(),
        'loss_ce': loss_ce.item(),
        'loss_cd': loss_cd.item(),
        'loss_rd': loss_rd.item(),
    }