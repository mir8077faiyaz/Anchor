"""
proto_etf_modules.py


Pipeline:
    HSI patch -> spectral encoder + spatial encoder -> bidirectional
    cross-attention -> invariant/private decomposition -> invariant
    refinement -> fixed ETF embedding.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralEncoder(nn.Module):
    """HSI-specific spectral encoder.

    Each spectral band is represented using two cheap statistics over the
    patch (mean and standard deviation), followed by a shared nonlinear
    band mixer. Contiguous band groups are pooled into tokens and refined by
    a Transformer. This preserves the original branch's stable group-token
    structure while giving each band local distribution information.

    Output: (B, num_spectral_tokens, dim)
    """
    def __init__(self, in_channels: int, num_spectral_tokens: int = 8,
                 dim: int = 64, depth: int = 2, heads: int = 4,
                 mlp_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_spectral_tokens = num_spectral_tokens
        bounds = torch.linspace(0, in_channels, num_spectral_tokens + 1).round().long()
        self.register_buffer("group_bounds", bounds, persistent=False)

        # Per-band descriptors: mean, std, and first-order spectral difference.
        self.band_mixer = nn.Sequential(
            nn.Linear(3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, num_spectral_tokens, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False,)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        mean = x.mean(dim=(2, 3))
        std = x.std(dim=(2, 3), unbiased=False)
        delta = torch.zeros_like(mean)
        if C > 1:
            delta[:, 1:] = mean[:, 1:] - mean[:, :-1]
        desc = torch.stack([mean, std, delta], dim=-1)  # B,C,3
        band_feat = self.band_mixer(desc)               # B,C,D

        tokens = []
        for i in range(self.num_spectral_tokens):
            lo = int(self.group_bounds[i].item())
            hi = int(self.group_bounds[i + 1].item())
            # Group mean + max gives both average and salient band response.
            g = band_feat[:, lo:hi]
            pooled = 0.5 * (g.mean(dim=1) + g.max(dim=1).values)
            tokens.append(pooled)
        tokens = torch.stack(tokens, dim=1)
        tokens = self.transformer(tokens + self.pos_embed)
        return self.norm(tokens)


class ResidualSpatialBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        p = dilation
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=p, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=p, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class SpatialEncoder(nn.Module):
    """Multi-scale spatial encoder with residual local context blocks."""
    def __init__(self, in_channels: int, patch_size: int, dim: int = 64,
                 conv_channels: int = 48):
        super().__init__()
        self.band_reduce = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, 1, bias=False),
            nn.BatchNorm2d(conv_channels),
            nn.GELU(),
        )
        self.local = ResidualSpatialBlock(conv_channels, dilation=1)
        self.context = ResidualSpatialBlock(conv_channels, dilation=2)
        self.down = nn.Sequential(
            nn.Conv2d(conv_channels, dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.refine = ResidualSpatialBlock(dim, dilation=1)
        out_hw = math.ceil(patch_size / 2)
        self.num_spatial_tokens = out_hw * out_hw
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_spatial_tokens, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.band_reduce(x)
        feat = self.local(feat)
        feat = self.context(feat)
        feat = self.down(feat)
        feat = self.refine(feat)
        B, D, Hp, Wp = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        return self.norm(tokens + self.pos_embed)


class CrossAttentionFusion(nn.Module):
    """Bidirectional spectral/spatial cross-attention fusion."""
    def __init__(self, dim: int = 64, heads: int = 4, proto_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.spec_to_spat = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.spat_to_spec = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.out_mlp = nn.Sequential(
            nn.Linear(dim * 2, proto_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(proto_dim, proto_dim), nn.LayerNorm(proto_dim),
        )

    def forward(self, spec_tokens, spat_tokens):
        A, _ = self.spec_to_spat(spec_tokens, spat_tokens, spat_tokens)
        A = self.norm_a(A + spec_tokens)
        B, _ = self.spat_to_spec(spat_tokens, spec_tokens, spec_tokens)
        B = self.norm_b(B + spat_tokens)
        pooled = torch.cat([A.mean(1), B.mean(1)], dim=-1)
        return self.out_mlp(pooled)


class InvariantPrivateDecomposition(nn.Module):
    """Lightweight TIP-inspired decomposition plus invariant refinement."""
    def __init__(self, proto_dim: int = 128, private_dim: int | None = None):
        super().__init__()
        private_dim = private_dim or proto_dim
        self.inv_proj = nn.Sequential(nn.Linear(proto_dim, proto_dim), nn.GELU(), nn.LayerNorm(proto_dim))
        self.priv_proj = nn.Sequential(nn.Linear(proto_dim, private_dim), nn.GELU(), nn.LayerNorm(private_dim))
        self.refine_gate = nn.Sequential(
            nn.Linear(proto_dim, proto_dim // 2), nn.GELU(),
            nn.Linear(proto_dim // 2, proto_dim), nn.Sigmoid(),
        )
        self.refine_proj = nn.Linear(proto_dim, proto_dim)
        self.refine_norm = nn.LayerNorm(proto_dim)

    def forward(self, latent):
        inv = self.inv_proj(latent)
        priv = self.priv_proj(latent)
        gate = self.refine_gate(inv)
        inv = self.refine_norm(inv + gate * self.refine_proj(inv))
        return inv, priv


class ProtoEncoder(nn.Module):
    """Complete V2 encoder. forward() returns invariant 128-D features."""
    def __init__(self, in_channels: int, patch_size: int, dim: int = 64,
                 proto_dim: int = 128, num_spectral_tokens: int = 8, heads: int = 4):
        super().__init__()
        self.spectral_encoder = SpectralEncoder(
            in_channels, num_spectral_tokens=num_spectral_tokens, dim=dim, heads=heads
        )
        self.spatial_encoder = SpatialEncoder(in_channels, patch_size, dim=dim)
        self.fusion = CrossAttentionFusion(dim=dim, heads=heads, proto_dim=proto_dim)
        self.decomposition = InvariantPrivateDecomposition(proto_dim=proto_dim, private_dim=proto_dim)

    def forward_features(self, x):
        spec = self.spectral_encoder(x)
        spat = self.spatial_encoder(x)
        latent = self.fusion(spec, spat)
        inv, priv = self.decomposition(latent)
        return {"latent": latent, "invariant": inv, "private": priv}

    def forward(self, x):
        return self.forward_features(x)["invariant"]


# --------------------------------------------------------------------------
# Fixed standard ETF simplex -- intentionally unchanged in geometry.
# --------------------------------------------------------------------------
def generate_etf_simplex(num_classes: int, feat_dim: int, seed: int = 0) -> torch.Tensor:
    assert feat_dim >= num_classes - 1
    g = torch.Generator().manual_seed(seed)
    rand_mat = torch.randn(feat_dim, num_classes, generator=g)
    U, _ = torch.linalg.qr(rand_mat)
    I = torch.eye(num_classes)
    ones = torch.ones(num_classes, num_classes)
    centering = I - ones / num_classes
    scale = math.sqrt(num_classes / (num_classes - 1))
    M = scale * (U @ centering)
    return F.normalize(M.t().contiguous(), dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 4, 103, 15, 15
    model = ProtoEncoder(C, H)
    x = torch.randn(B, C, H, W)
    f = model.forward_features(x)
    print({k: tuple(v.shape) for k, v in f.items()})
    etf = generate_etf_simplex(9, 128, seed=2024)
    sims = etf @ etf.t()
    mask = ~torch.eye(9, dtype=torch.bool)
    print("ETF off-diagonal mean:", sims[mask].mean().item(), "expected:", -1/8)
