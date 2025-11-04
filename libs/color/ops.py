# libs/color/ops.py
from __future__ import annotations
import numpy as np
import torch

import colour  # type: ignore
from colour.models import (
    RGB_Colourspace,
    log_decoding_ACEScct,
    log_encoding_ACEScct,
    normalised_primary_matrix,
    oetf_BT709,
    oetf_inverse_BT709,
)
from colour.models import (
    log_decoding_ARRILogC3,
    log_encoding_ARRILogC3,
)

if "ARRI Wide Gamut 3" not in colour.RGB_COLOURSPACES:
    AWG3_PRIMARIES = np.array([
        [0.7347, 0.2653],
        [0.1426, 0.8574],
        [0.0991, -0.0308],
    ])
    AWG3_WHITEPOINT_NAME = "D65"
    AWG3_WHITEPOINT = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][AWG3_WHITEPOINT_NAME]
    matrix_rgb_to_xyz = normalised_primary_matrix(AWG3_PRIMARIES, AWG3_WHITEPOINT)
    matrix_xyz_to_rgb = np.linalg.inv(matrix_rgb_to_xyz)
    colour.RGB_COLOURSPACES["ARRI Wide Gamut 3"] = RGB_Colourspace(
        "ARRI Wide Gamut 3",
        AWG3_PRIMARIES,
        AWG3_WHITEPOINT,
        AWG3_WHITEPOINT_NAME,
        matrix_rgb_to_xyz,
        matrix_xyz_to_rgb,
        cctf_encoding=None,
        cctf_decoding=None,
    )

_COLOURSPACE_OBJECTS = {
    "AWG3": colour.RGB_COLOURSPACES["ARRI Wide Gamut 3"],
    "ACESCG": colour.RGB_COLOURSPACES["ACEScg"],
    "REC709": colour.RGB_COLOURSPACES["Rec. 709"],
}


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


def _logc3_to_linear_awg3(img: np.ndarray) -> np.ndarray:
    logc = np.clip(img.astype(np.float64, copy=False), 0.0, 1.0)
    lin = log_decoding_ARRILogC3(logc)
    return np.clip(lin.astype(np.float64, copy=False), 0.0, None)


def _linear_awg3_to_logc3(img: np.ndarray) -> np.ndarray:
    linear = np.clip(img.astype(np.float64, copy=False), 0.0, None)
    logc = log_encoding_ARRILogC3(linear)
    return np.clip(logc.astype(np.float64, copy=False), 0.0, 1.0)


def _decode_to_linear(img: np.ndarray, space: str) -> tuple[np.ndarray, str]:
    space_key = space.replace("-", "").replace(" ", "").upper()
    if space_key == "LOGC3":
        return _logc3_to_linear_awg3(img), "AWG3"
    if space_key == "AWG3":
        return img.astype(np.float64, copy=False), "AWG3"
    if space_key == "ACESCG":
        return img.astype(np.float64, copy=False), "ACESCG"
    if space_key == "ACESCCT":
        lin = log_decoding_ACEScct(np.clip(img, 0.0, 1.0))
        return lin.astype(np.float64, copy=False), "ACESCG"
    if space_key in ("REC709", "BT709"):
        lin = oetf_inverse_BT709(np.clip(img, 0.0, 1.0))
        return lin.astype(np.float64, copy=False), "REC709"
    raise NotImplementedError(f"Unsupported source space: {space}")


def _encode_from_linear(img: np.ndarray, space: str) -> np.ndarray:
    space_key = space.replace("-", "").replace(" ", "").upper()
    if space_key == "LOGC3":
        return _linear_awg3_to_logc3(img).astype(np.float32, copy=False)
    if space_key == "AWG3":
        return np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
    if space_key == "ACESCG":
        return np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
    if space_key == "ACESCCT":
        enc = log_encoding_ACEScct(np.clip(img, 0.0, None))
        return np.clip(enc, 0.0, 1.0).astype(np.float32, copy=False)
    if space_key in ("REC709", "BT709"):
        enc = oetf_BT709(np.clip(img, 0.0, None))
        return np.clip(enc, 0.0, 1.0).astype(np.float32, copy=False)
    raise NotImplementedError(f"Unsupported destination space: {space}")


def _convert_linear_space(rgb: np.ndarray, src_space: str, dst_space: str) -> np.ndarray:
    src_key = src_space.upper()
    dst_key = dst_space.upper()
    if src_key == dst_key:
        return rgb
    try:
        src_cs = _COLOURSPACE_OBJECTS[src_key]
        dst_cs = _COLOURSPACE_OBJECTS[dst_key]
    except KeyError as exc:
        raise NotImplementedError(f"Unsupported colourspace transform: {src_space} -> {dst_space}") from exc
    shp = rgb.shape
    flat = rgb.reshape(-1, 3)
    converted = colour.RGB_to_RGB(
        flat,
        src_cs,
        dst_cs,
        apply_cctf_decoding=False,
        apply_cctf_encoding=False,
    )
    return converted.reshape(shp)


def convert_space(img: np.ndarray, src: str, dst: str, meta=None) -> np.ndarray:
    """
    Unified colour/EOTF conversion entry.
    Supports:
      - LogC3 (ARRI) <-> AWG3 linear
      - AWG3 linear <-> ACEScg / ACEScct
      - AWG3 linear <-> Rec.709 (BT.709)
    """
    if src == dst:
        return img

    linear_src, src_linear_space = _decode_to_linear(img, src)
    dst_linear_space = src_linear_space
    if dst.replace("-", "").replace(" ", "").upper() not in ("LOGC3", src_linear_space):
        if dst.replace("-", "").replace(" ", "").upper() == "ACESCCT":
            dst_linear_space = "ACESCG"
        elif dst.replace("-", "").replace(" ", "").upper() in ("REC709", "BT709"):
            dst_linear_space = "REC709"
        elif dst.replace("-", "").replace(" ", "").upper() == "ACESCG":
            dst_linear_space = "ACESCG"
        elif dst.replace("-", "").replace(" ", "").upper() == "LOGC3":
            dst_linear_space = "AWG3"
        else:
            raise NotImplementedError(f"Unsupported destination space: {dst}")

    converted_linear = _convert_linear_space(linear_src, src_linear_space, dst_linear_space)
    encoded = _encode_from_linear(converted_linear, dst)
    return encoded
