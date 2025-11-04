# libs/color/lut.py
from __future__ import annotations
import numpy as np


def load_cube_lut(path: str) -> dict:
    """
    Parse a 3D .cube file.
    Returns:
      {
        "table": np.ndarray[L, L, L, 3] in [0,1],
        "domain_min": np.ndarray[3],
        "domain_max": np.ndarray[3],
      }
    """
    size = None
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    data = []

    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            key = toks[0].upper()
            if key == "TITLE":
                continue
            if key == "LUT_3D_SIZE":
                size = int(toks[1])
                continue
            if key == "DOMAIN_MIN":
                domain_min = np.array(list(map(float, toks[1:4])), dtype=np.float32)
                continue
            if key == "DOMAIN_MAX":
                domain_max = np.array(list(map(float, toks[1:4])), dtype=np.float32)
                continue
            # data row (R G B)
            if len(toks) >= 3:
                data.append(list(map(float, toks[:3])))

    if size is None:
        raise NotImplementedError("Only 3D .cube with LUT_3D_SIZE is supported.")

    data = np.asarray(data, dtype=np.float32)
    expected = size * size * size
    if data.shape[0] != expected:
        raise ValueError(f"LUT entries mismatch: got {data.shape[0]}, expect {expected}")

    table = data.reshape(size, size, size, 3)
    table = np.clip(table, 0.0, 1.0)
    return {
        "table": table,
        "domain_min": domain_min,
        "domain_max": domain_max,
    }


def _trilinear_sample(table: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    table: [N,N,N,3], xyz: [...,3] in [0,1] -> [...,3]
    """
    N = table.shape[0]
    p = xyz * (N - 1)
    i0 = np.floor(p).astype(np.int32)
    d = p - i0
    i1 = np.clip(i0 + 1, 0, N - 1)

    x0, y0, z0 = i0[..., 0], i0[..., 1], i0[..., 2]
    x1, y1, z1 = i1[..., 0], i1[..., 1], i1[..., 2]
    dx, dy, dz = d[..., 0:1], d[..., 1:2], d[..., 2:3]

    c000 = table[x0, y0, z0]
    c100 = table[x1, y0, z0]
    c010 = table[x0, y1, z0]
    c110 = table[x1, y1, z0]
    c001 = table[x0, y0, z1]
    c101 = table[x1, y0, z1]
    c011 = table[x0, y1, z1]
    c111 = table[x1, y1, z1]

    c00 = c000 * (1 - dx) + c100 * dx
    c01 = c001 * (1 - dx) + c101 * dx
    c10 = c010 * (1 - dx) + c110 * dx
    c11 = c011 * (1 - dx) + c111 * dx
    c0 = c00 * (1 - dy) + c10 * dy
    c1 = c01 * (1 - dy) + c11 * dy
    c = c0 * (1 - dz) + c1 * dz
    return c


def apply_3d_lut(img: np.ndarray, lut: dict, mode: str = "trilinear") -> np.ndarray:
    """
    Apply 3D LUT to image in [0,1].
    """
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError("img must be HxWx3")
    table = lut["table"]
    dmin = lut["domain_min"]
    dmax = lut["domain_max"]

    x = (img - dmin) / (dmax - dmin + 1e-8)
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)

    if mode == "nearest":
        N = table.shape[0]
        p = np.round(x * (N - 1)).astype(np.int32)
        out = table[p[..., 0], p[..., 1], p[..., 2]]
    else:
        out = _trilinear_sample(table, x)

    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
