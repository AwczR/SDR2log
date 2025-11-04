# libs/color/ops.py
from __future__ import annotations
import numpy as np
import torch


def ensure_even_hw(img: np.ndarray, how: str = "center_crop") -> np.ndarray:
    """
    Ensure H and W are even.
    - center_crop: crop 1px borders symmetrically.
    - floor: truncate to the nearest even size (top-left).
    """
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError("img must be HxWx3")
    H, W = img.shape[:2]
    H2 = H - (H % 2)
    W2 = W - (W % 2)
    if H2 == H and W2 == W:
        return img
    if how == "floor":
        return np.ascontiguousarray(img[:H2, :W2, :])
    # center_crop
    top = (H - H2) // 2
    left = (W - W2) // 2
    return np.ascontiguousarray(img[top:top + H2, left:left + W2, :])


def to_chw_tensor(img: np.ndarray) -> torch.FloatTensor:
    """
    Convert HxWx3 float32 [0,1] to torch.FloatTensor 3xHxW (contiguous).
    """
    if img.dtype != np.float32:
        img = img.astype(np.float32, copy=False)
    t = torch.from_numpy(np.transpose(img, (2, 0, 1))).contiguous()
    return t  # FloatTensor


def convert_space(img: np.ndarray, src: str, dst: str, meta=None) -> np.ndarray:
    """
    Unified color/EOTF conversion entry.
    Current minimal behavior:
      - if src == dst: identity
      - else: NotImplementedError
    Extend here with:
      - LogC3/AWG3 <-> ACEScct/ACEScg
      - HLG/PQ <-> ACEScct
      - Rec709 <-> ACEScct
    """
    if src == dst:
        return img
    raise NotImplementedError(f"convert_space not implemented: {src} -> {dst}")
