# model.py
# -*- coding: utf-8 -*-
"""
Diffusion-based SDRTV-to-HDRTV Reconstruction with Artifact Suppression (Diff-HDRTV)
with two FAIR modes (same #branches, similar parameterization):

  - mode="xyY": 论文原设（Stage-1: HDRNet -> Stage-2: DiffusionPrior(DDIM) -> Stage-3: xy-Net + Y-Net + HDR-xyY-Net）
  - mode="aces": 结构对等，但全程在 ACEScct 域。使用一个可学习的线性分解将 ACEScct 的 3 通道拆成
                 "色度C(2ch) + 亮度L(1ch)"，然后用 C-Net / L-Net / ACES-Refiner 三分支与先验融合。

===========
I/O SPEC
===========

Tensor convention:
    - dtype: float32, range [0, 1], shape [B, 3, H, W]
    - H, W must be divisible by 2  (Stage-1: 7x7, s=2; PixelShuffle upsample)

mode="xyY"（论文原设）：
    Input  (SDR RGB):    x_sdr_rgb [B,3,H,W]
    Output (HDR RGB):    y_hdr_rgb [B,3,H,W]
    Interm:
        - hdr_ref   (RGB)   [B,3,H,W]  —— Stage-1 输出
        - hdr_prior (RGB)   [B,3,H,W]  —— Stage-2 扩散先验（内部低分辨率生成，后上采样）

mode="aces"（公平对比，结构等价）：
    Input  (SDR ACEScct):  x_sdr_aces [B,3,H,W]   （数据集已在 ACEScct）
    Output (HDR ACEScct):  y_hdr_aces [B,3,H,W]
    Interm:
        - hdr_ref   (ACES)  [B,3,H,W]
        - hdr_prior (ACES)  [B,3,H,W]

Stage-2（两种模式相同）内部默认在 low_res=(128,256) 生成先验（可配置），再上采样回原分辨率。

Dependency for Stage-2:
    pip install diffusers>=0.30.0 accelerate

Color conversion:
    - 为避免把“色彩空间矩阵/EOTF 假设”硬编码到网络里，xyY<->RGB 的精确实现建议放到数据管线。
    - 这里仅给 minimal 的占位实现（基于 colour-science 的 sRGB 矩阵）。实际训练时请替换为你的 BT.2020/PQ 等流程。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional dependencies for Stage-2
try:
    from diffusers import UNet2DModel, DDIMScheduler
    _HAS_DIFFUSERS = True
except Exception:
    _HAS_DIFFUSERS = False

# Optional colour conversion helpers (xyY <-> RGB) — minimal placeholders
try:
    import colour  # from 'colour-science'
    _HAS_COLOUR = True
except Exception:
    _HAS_COLOUR = False


# -----------------------------
# Utilities
# -----------------------------

def _check_size(x: torch.Tensor):
    h, w = x.shape[-2:]
    if (h % 2) or (w % 2):
        raise ValueError(f"H and W must be divisible by 2. Got H={h}, W={w}.")

def _resize_to(x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size_hw, mode="bicubic", align_corners=False)

def _concat_channels(*xs: torch.Tensor) -> torch.Tensor:
    return torch.cat(xs, dim=1)

# --- xyY helpers (建议在数据管线中替换为真实 BT.2020/PQ 矩阵) ---
def _rgb_to_xyY(rgb: torch.Tensor, clip: bool = True) -> torch.Tensor:
    if not _HAS_COLOUR:
        raise RuntimeError("colour-science not installed. Install 'colour-science' or handle xyY conversion in your data pipeline.")
    b, c, h, w = rgb.shape
    rgb_np = rgb.permute(0, 2, 3, 1).contiguous().view(-1, 3).cpu().numpy()
    XYZ = colour.RGB_to_XYZ(
        rgb_np,
        colour.RGB_COLOURSPACES["sRGB"].whitepoint,
        colour.RGB_COLOURSPACES["sRGB"].whitepoint,
        colour.RGB_COLOURSPACES["sRGB"].matrix_RGB_to_XYZ,
    )
    xyY = colour.XYZ_to_xyY(XYZ)
    xyY = torch.from_numpy(xyY).to(rgb.device, dtype=rgb.dtype).view(b, h, w, 3).permute(0, 3, 1, 2)
    if clip:
        xyY = torch.clamp(xyY, 0.0, 1.0)
    return xyY

def _xyY_to_rgb(xyY: torch.Tensor, clip: bool = True) -> torch.Tensor:
    if not _HAS_COLOUR:
        raise RuntimeError("colour-science not installed. Install 'colour-science' or handle xyY conversion in your data pipeline.")
    b, c, h, w = xyY.shape
    xyY_np = xyY.permute(0, 2, 3, 1).contiguous().view(-1, 3).cpu().numpy()
    XYZ = colour.xyY_to_XYZ(xyY_np)
    rgb = colour.XYZ_to_RGB(
        XYZ,
        colour.RGB_COLOURSPACES["sRGB"].whitepoint,
        colour.RGB_COLOURSPACES["sRGB"].whitepoint,
        colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB,
    )
    rgb = torch.from_numpy(rgb).to(xyY.device, dtype=xyY.dtype).view(b, h, w, 3).permute(0, 3, 1, 2)
    if clip:
        rgb = torch.clamp(rgb, 0.0, 1.0)
    return rgb


# -----------------------------
# Stage-1: HDRNet (论文 §3.2)
# -----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=k // 2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

class ResBlockNoBN(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.c1(x))
        out = self.c2(out)
        return x + out

class HDRNet(nn.Module):
    """
    ResBlock(x) = x + Conv3x3_c64 ∘ ReLU ∘ Conv3x3_c64 (x)
    HDRNet(x)   = Conv3x3_c64 ∘ PixelShuffle ∘ ResBlock×3 ∘ Conv3x3_c64 ∘ Conv7x7_s2_c128 (x)
    无 BN，PixelShuffle 还原到输入分辨率。
    """
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.head = ConvBlock(in_ch, 128, k=7, s=2)   # 7x7, s=2
        self.down_to_64 = ConvBlock(128, 64, k=3, s=1)
        self.body = nn.Sequential(ResBlockNoBN(64), ResBlockNoBN(64), ResBlockNoBN(64))
        self.up = nn.Sequential(
            nn.Conv2d(64, 64 * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(64, out_ch, 3, 1, 1)

    def forward(self, x):
        _check_size(x)
        x = self.head(x)
        x = self.down_to_64(x)
        x = self.body(x)
        x = self.up(x)
        x = self.tail(x)
        return torch.clamp(x, 0.0, 1.0)


# -----------------------------
# Stage-2: Diffusion Prior (DDIM)
# -----------------------------

@dataclass
class DiffusionCfg:
    low_res: Tuple[int, int] = (128, 256)  # (H, W)
    num_inference_steps: int = 50
    noise_std: float = 0.1

class DiffusionPrior(nn.Module):
    """
    使用 UNet2DModel + DDIMScheduler。
    初始化不是纯噪声：对 HDR_ref 下采样并加小噪声，从而更接近真实数据流形。
    """
    def __init__(self, in_ch=3, base_channels=64, cfg: DiffusionCfg = DiffusionCfg()):
        super().__init__()
        if not _HAS_DIFFUSERS:
            raise RuntimeError("`diffusers` is required for Stage-2. Install with `pip install diffusers`.")
        self.cfg = cfg

        self.unet = UNet2DModel(
            sample_size=cfg.low_res[0],  # 运行时会做 HxW 插值，这里给出一个尺寸参考
            in_channels=in_ch,
            out_channels=in_ch,
            block_out_channels=(base_channels, base_channels*2, base_channels*4),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
            layers_per_block=2,
        )

        self.scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear",
            clip_sample=False, set_alpha_to_one=False
        )

    @torch.no_grad()
    def forward(self, hdr_ref: torch.Tensor) -> torch.Tensor:
        b, c, h, w = hdr_ref.shape
        lr = _resize_to(hdr_ref, self.cfg.low_res)
        x_t = torch.clamp(lr + torch.randn_like(lr) * self.cfg.noise_std, 0.0, 1.0)

        self.scheduler.set_timesteps(self.cfg.num_inference_steps, device=hdr_ref.device)
        for t in self.scheduler.timesteps:
            eps = self.unet(x_t, t).sample
            x_t = self.scheduler.step(eps, t, x_t).prev_sample

        hdr_prior = _resize_to(x_t, (h, w))
        return torch.clamp(hdr_prior, 0.0, 1.0)


# -----------------------------
# Stage-3: Refiner (xyY) —— 论文原设
# -----------------------------

class XYNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = HDRNet(in_ch=2, out_ch=2)
    def forward(self, sdr_xy):
        return torch.clamp(self.backbone(sdr_xy), 0.0, 1.0)

class YNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = HDRNet(in_ch=1, out_ch=1)
    def forward(self, sdr_y):
        return torch.clamp(self.backbone(sdr_y), 0.0, 1.0)

class XYRefiner(nn.Module):
    """
    HDR-xyY-Net: concat(HDR_xy, HDR_Y, HDR_prior_rgb) -> final HDR RGB
    """
    def __init__(self, in_ch=3+2+1, out_ch=3):
        super().__init__()
        self.backbone = HDRNet(in_ch=in_ch, out_ch=out_ch)
    def forward(self, hdr_xy, hdr_y, hdr_prior_rgb):
        x = _concat_channels(hdr_xy, hdr_y, hdr_prior_rgb)
        return torch.clamp(self.backbone(x), 0.0, 1.0)


# -----------------------------
# Stage-3: Refiner (ACES) —— 结构对等、空间不同
# -----------------------------

class ACESDecomposer(nn.Module):
    """
    将 ACEScct 的 3 通道通过一个“可学习的线性分解”得到：
        L (1ch)  —— “亮度”样的通道
        C (2ch)  —— “色度”样的两个互补通道
    为公平起见，保持与 xyY 模式相同的通道数分配(2 + 1)。

    实现：1x1 Conv（不带偏置）投影到 3 维，再拆成 (2,1)。
    - 初始化：L 通道权重用常见亮度系数近似（如 [0.2126, 0.7152, 0.0722]，可自行替换/训练中自适应）。
    - C 通道用一个 2x3 的正交补近似初始化（随机归一化），训练可自更新。
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            # 初始化第三行作为 L（亮度）投影系数（可按需要替换为 ACES/BT.2020 对应权重）
            L = torch.tensor([0.2126, 0.7152, 0.0722])  # 近似 Rec.709 luma
            # 随机生成两个与 L 近似正交的向量作为 C 的基（简单正交化）
            R = torch.randn(2, 3)
            # Gram-Schmidt 去投影
            Ln = L / (L.norm() + 1e-8)
            for i in range(2):
                R[i] = R[i] - (R[i] @ Ln) * Ln
            # 归一化
            for i in range(2):
                n = R[i].norm()
                if n > 0:
                    R[i] = R[i] / n
            W = torch.vstack([R, L]).float()  # [3,3]，前两行->C，最后一行->L
            self.proj.weight.copy_(W.unsqueeze(-1).unsqueeze(-1))  # [3,3,1,1]

    def forward(self, x_aces: torch.Tensor) -> Dict[str, torch.Tensor]:
        y = self.proj(x_aces)           # [B,3,H,W]
        C = y[:, :2, ...]               # [B,2,H,W]
        L = y[:, 2:, ...]               # [B,1,H,W]
        return {"C": C, "L": L}

class CNet(nn.Module):
    """C-Net: maps SDR-ACES chroma-like 2ch -> HDR-ACES chroma-like 2ch"""
    def __init__(self):
        super().__init__()
        self.backbone = HDRNet(in_ch=2, out_ch=2)
    def forward(self, sdr_C):
        return torch.clamp(self.backbone(sdr_C), 0.0, 1.0)

class LNet(nn.Module):
    """L-Net: maps SDR-ACES luma-like 1ch -> HDR-ACES luma-like 1ch"""
    def __init__(self):
        super().__init__()
        self.backbone = HDRNet(in_ch=1, out_ch=1)
    def forward(self, sdr_L):
        return torch.clamp(self.backbone(sdr_L), 0.0, 1.0)

class ACESRefiner3Branch(nn.Module):
    """
    与 xyY 模式对等的融合头：
        concat(HDR_C(2), HDR_L(1), HDR_prior_aces(3)) -> final HDR ACEScct (3)
    """
    def __init__(self, in_ch=2+1+3, out_ch=3):
        super().__init__()
        self.backbone = HDRNet(in_ch=in_ch, out_ch=out_ch)
    def forward(self, hdr_C, hdr_L, hdr_prior_aces):
        x = _concat_channels(hdr_C, hdr_L, hdr_prior_aces)
        return torch.clamp(self.backbone(x), 0.0, 1.0)


# -----------------------------
# Full Model (two modes)
# -----------------------------

@dataclass
class ModelCfg:
    mode: Literal["xyY", "aces"] = "xyY"
    diffusion: DiffusionCfg = DiffusionCfg()

class DiffHDRTV(nn.Module):
    """
    mode="xyY":
        x_sdr_rgb -> HDRNet -> HDR_ref_rgb
            -> DiffusionPrior -> HDR_prior_rgb
            -> xy conversion: (x->xyY) -> xy-Net/Y-Net
            -> HDR-xyY-Net concat(hdr_xy, hdr_Y, hdr_prior_rgb) -> y_hdr_rgb

    mode="aces":
        x_sdr_acescct -> HDRNet -> HDR_ref_aces
            -> DiffusionPrior -> HDR_prior_aces
            -> ACESDecomposer: x -> (C,L)
            -> C-Net / L-Net
            -> ACES-Refiner-3Branch concat(hdr_C, hdr_L, hdr_prior_aces) -> y_hdr_aces
    """
    def __init__(self, cfg: ModelCfg = ModelCfg()):
        super().__init__()
        self.cfg = cfg
        self.mode = cfg.mode

        # Stage-1 (共享)
        self.hdrnet = HDRNet(in_ch=3, out_ch=3)

        # Stage-2 (共享)
        self.prior = DiffusionPrior(in_ch=3, base_channels=64, cfg=cfg.diffusion)

        # Stage-3
        if self.mode == "xyY":
            self.xynet = XYNet()
            self.ynet = YNet()
            self.refiner_xyY = XYRefiner(in_ch=3 + 2 + 1, out_ch=3)
        elif self.mode == "aces":
            self.decomp = ACESDecomposer()
            self.cnet = CNet()
            self.lnet = LNet()
            self.refiner_aces3 = ACESRefiner3Branch(in_ch=2 + 1 + 3, out_ch=3)
        else:
            raise ValueError("cfg.mode must be 'xyY' or 'aces'.")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B,3,H,W] in [0,1]
            - mode="xyY":  SDR RGB
            - mode="aces": SDR ACEScct
        returns:
            - "hdr_ref"   : [B,3,H,W]
            - "hdr_prior" : [B,3,H,W]
            - "y"         : [B,3,H,W]  (HDR RGB for xyY mode; HDR ACEScct for aces mode)
        """
        _check_size(x)

        # Stage-1
        hdr_ref = self.hdrnet(x)

        # Stage-2
        hdr_prior = self.prior(hdr_ref).detach()

        if self.mode == "xyY":
            sdr_xyY = _rgb_to_xyY(x)  # 占位：训练时请替换为你的真实管线
            sdr_xy, sdr_Y = sdr_xyY[:, :2, ...], sdr_xyY[:, 2:, ...]
            hdr_xy = self.xynet(sdr_xy)
            hdr_Y  = self.ynet(sdr_Y)
            y_rgb  = self.refiner_xyY(hdr_xy, hdr_Y, hdr_prior)
            return {"hdr_ref": hdr_ref, "hdr_prior": hdr_prior, "y": y_rgb}

        else:  # "aces"
            comp = self.decomp(x)     # {C: [B,2,H,W], L: [B,1,H,W]}
            hdr_C = self.cnet(comp["C"])
            hdr_L = self.lnet(comp["L"])
            y_aces = self.refiner_aces3(hdr_C, hdr_L, hdr_prior)
            return {"hdr_ref": hdr_ref, "hdr_prior": hdr_prior, "y": y_aces}


# -----------------------------
# Factory
# -----------------------------

def build_model(
    mode: Literal["xyY", "aces"] = "xyY",
    low_res: Tuple[int, int] = (128, 256),
    num_inference_steps: int = 50,
    noise_std: float = 0.1,
) -> DiffHDRTV:
    cfg = ModelCfg(
        mode=mode,
        diffusion=DiffusionCfg(
            low_res=low_res,
            num_inference_steps=num_inference_steps,
            noise_std=noise_std,
        ),
    )
    return DiffHDRTV(cfg)
