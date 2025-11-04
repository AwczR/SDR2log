# libs/color/io.py
from __future__ import annotations
import os
import numpy as np
import tifffile


def _normalize_to_01(arr: np.ndarray) -> np.ndarray:
    """Map image array to float32 [0,1]. Supports integer and float inputs."""
    if np.issubdtype(arr.dtype, np.floating):
        out = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
        return out
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        out = (arr.astype(np.float32) / float(info.max))
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported dtype for normalization: {arr.dtype}")


def read_tiff_as_float01(path: str) -> np.ndarray:
    """
    Read .tif/.tiff into float32 RGB in [0,1].
    - Grayscale -> replicate to 3 channels.
    - RGBA -> drop alpha.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    img = tifffile.imread(path)  # HxW or HxWxC
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim != 3 or img.shape[-1] not in (3, 4):
        raise ValueError(f"Unsupported TIFF shape: {img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = _normalize_to_01(img)
    return np.ascontiguousarray(img)
