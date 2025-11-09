"""Attention U-Net baseline for SDR-to-HDR regression in ACEScct space."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Additive attention gate as in Attention U-Net."""

    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.theta = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.phi = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        attn = self.act(self.theta(x) + self.phi(g))
        attn = self.psi(attn)
        return x * attn


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attn = AttentionGate(out_ch, skip_ch, inter_ch=max(out_ch // 2, 1))
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.attn(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, features: Sequence[int] = (64, 128, 256, 512)):
        super().__init__()
        if len(features) < 2:
            raise ValueError("Need at least two levels for U-Net")
        chs = list(features)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_ch
        for ch in chs:
            self.encoders.append(DoubleConv(prev, ch))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev = ch
        self.bottleneck = DoubleConv(prev, prev * 2)

        dec_channels = chs[::-1]
        self.upblocks = nn.ModuleList()
        cur = prev * 2
        for idx, ch in enumerate(dec_channels):
            out_ch_block = ch
            self.upblocks.append(UpBlock(cur, ch, out_ch_block))
            cur = out_ch_block
        self.tail = nn.Conv2d(cur, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        out = x
        for encoder, pool in zip(self.encoders, self.pools):
            out = encoder(out)
            skips.append(out)
            out = pool(out)
        out = self.bottleneck(out)
        for up, skip in zip(self.upblocks, reversed(skips)):
            out = up(out, skip)
        return self.tail(out)


@dataclass
class ModelCfg:
    in_channels: int = 3
    out_channels: int = 3
    features: Iterable[int] = (64, 128, 256, 512)
    final_activation: str = "sigmoid"  # "sigmoid", "tanh", or "none"


class SDR2HDRAttentionUNet(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.unet = AttentionUNet(cfg.in_channels, cfg.out_channels, tuple(cfg.features))
        activation = cfg.final_activation.lower()
        if activation not in {"sigmoid", "tanh", "none"}:
            raise ValueError(f"Unsupported final_activation: {cfg.final_activation}")
        self.activation = activation

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.unet(x)
        if self.activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.activation == "tanh":
            out = (torch.tanh(out) + 1.0) * 0.5
        return {"pred": out}


def build_model(**kwargs) -> SDR2HDRAttentionUNet:
    cfg = ModelCfg(**kwargs)
    return SDR2HDRAttentionUNet(cfg)
