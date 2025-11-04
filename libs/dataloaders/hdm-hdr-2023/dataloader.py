# lib/dataloaders/hdm-hdr-2023/dataloader.py
import os
import glob
import random
from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 颜色与I/O统一从 libs/color 走
from libs.color import (
    read_tiff_as_float01,
    load_cube_lut, apply_3d_lut,
    ensure_even_hw, to_chw_tensor, convert_space
)

__all__ = ["build_dataloader"]

class ImgPairFromLogC3(Dataset):
    """
    从 LogC3/AWG3 的 TIFF 构造 (sdr, hdr) 图像对（img-only）。
    - hdr: 以原始 LogC3 为“目标”，或经 convert_space 到指定训练空间。
    - sdr: 通过 3D LUT (LogC3->HLG/PQ/自定义) 或内置色彩转换 (LogC3->Rec709) 生成“伪SDR”。
    """

    def __init__(self, cfg: Dict[str, Any], split: str):
        super().__init__()
        data_cfg = cfg.get("data", {})
        self.root = data_cfg.get("root")
        if not self.root or not os.path.isdir(self.root):
            raise FileNotFoundError(f"root not found: {self.root}")

        # 切分规则：简单按文件名排序 + 固定比例/列表
        self.split = split
        self.train_ratio = float(data_cfg.get("train_ratio", 0.9))
        all_tiffs = sorted(
            glob.glob(os.path.join(self.root, "*.tif")) +
            glob.glob(os.path.join(self.root, "*.tiff"))
        )
        if len(all_tiffs) == 0:
            raise FileNotFoundError(f"No TIFF found under {self.root}")

        # 简单 split：前 90% 训练，其余验证/测试（若你有 index 文件可在此替换）
        n = len(all_tiffs)
        n_train = int(round(n * self.train_ratio))
        if split == "train":
            self.paths = all_tiffs[:n_train]
        elif split in ("val", "test"):
            self.paths = all_tiffs[n_train:]
        else:
            raise ValueError(f"Invalid split: {split}")

        # 选择“伪SDR”来源
        sdr_from = data_cfg.get("sdr_from", "hlg")  # "hlg" | "pq" | "custom" | "rec709"
        lut_dir = data_cfg.get("lut_dir", os.path.join(self.root, "LUTs_for_conversion_from_LogCv3-Camera-Footage_to_HLG_and_PQ"))
        self.space_in = data_cfg.get("space_in", "LogC3")
        self.space_out = data_cfg.get("space_out", self.space_in)  # 若暂未实现 ACEScct 变换，保持一致更稳妥
        valid_spaces = {"LogC3", "ACEScct", "ACEScg", "Rec709"}
        if self.space_in not in valid_spaces:
            raise ValueError(f"Unsupported space_in: {self.space_in}")
        if self.space_out not in valid_spaces:
            raise ValueError(f"Unsupported space_out: {self.space_out}")
        self.sdr_use_lut = True

        if sdr_from == "hlg":
            lut_path = os.path.join(lut_dir, "ARRI_LogC3-to-HLG_1K_Rec2100-D65_DW200_v2_65.cube")
        elif sdr_from == "pq":
            lut_path = os.path.join(lut_dir, "ARRI_LogC3-to-St2084_4K_Rec2100-D65_DW200_v2_65.cube")
        elif sdr_from == "custom":
            lut_path = data_cfg.get("sdr_lut_path", "")
        elif sdr_from == "rec709":
            lut_path = ""
            self.sdr_use_lut = False
        else:
            raise ValueError(f"Invalid sdr_from: {sdr_from}")

        if self.sdr_use_lut:
            if not lut_path or not os.path.isfile(lut_path):
                raise FileNotFoundError(f"LUT not found: {lut_path}")
            self.sdr_lut = load_cube_lut(lut_path)
        else:
            self.sdr_lut = None
        self.sdr_from = sdr_from
        self.sdr_lut_path = lut_path

        # 增强配置（可选）
        aug = data_cfg.get("augment", {})
        self.train_crop = tuple(aug.get("train_crop_size", [])) if split == "train" else None
        self.hflip = bool(aug.get("hflip", False)) if split == "train" else False

        # 随机数
        seed = int(cfg.get("seed", 42))
        self.rng = random.Random(seed + (0 if split=="train" else 777))

    def __len__(self):
        return len(self.paths)

    def _random_crop_even(self, img: np.ndarray, crop: tuple):
        """中心或随机裁剪为偶数尺寸；img: [H,W,3]"""
        H, W = img.shape[:2]
        ch, cw = crop
        ch = ch - (ch % 2)
        cw = cw - (cw % 2)
        if ch <= 0 or cw <= 0 or ch > H or cw > W:
            # 回退到确保偶数
            return ensure_even_hw(img, how="center_crop")
        if self.split == "train":
            top = self.rng.randint(0, H - ch)
            left = self.rng.randint(0, W - cw)
        else:
            top = (H - ch) // 2
            left = (W - cw) // 2
        return img[top:top+ch, left:left+cw, :]

    def _maybe_hflip(self, a: np.ndarray, b: np.ndarray):
        if self.hflip and self.rng.random() < 0.5:
            return np.ascontiguousarray(a[:, ::-1, :]), np.ascontiguousarray(b[:, ::-1, :])
        return a, b

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        # 1) 读 LogC3 tiff 到 [0,1]
        img_logc = read_tiff_as_float01(path)  # [H,W,3], float32 in [0,1]

        # 2) sdr = LUT 或色彩转换生成的伪SDR
        if self.sdr_use_lut:
            sdr_img = apply_3d_lut(img_logc, self.sdr_lut, mode="trilinear")
            sdr_src_space = "HLG" if self.sdr_from == "hlg" else "PQ" if self.sdr_from == "pq" else self.space_in
            lut_meta = os.path.basename(self.sdr_lut_path)
        else:
            sdr_img = convert_space(img_logc, src=self.space_in, dst="Rec709", meta={"path": path, "mode": "builtin"})
            sdr_src_space = "Rec709"
            lut_meta = "builtin_LogC3_to_Rec709"

        # 3) 可选色彩统一到 space_out（当前若 convert_space 未实现，将回退 identity）
        try:
            hdr_img = convert_space(img_logc, src=self.space_in, dst=self.space_out, meta={"path": path})
        except NotImplementedError:
            hdr_img = img_logc  # 暂时保留在 LogC3

        try:
            sdr_img = convert_space(sdr_img, src=sdr_src_space,
                                    dst=self.space_out, meta={"src": sdr_src_space, "transform": lut_meta})
        except NotImplementedError:
            # 若未实现，则维持渲染域标签
            pass

        # 4) 尺寸规范：偶数；可选随机裁剪/翻转（对齐做在 sdr/hdr 同步上）
        if self.train_crop is not None:
            hdr_img = self._random_crop_even(hdr_img, self.train_crop)
            sdr_img = self._random_crop_even(sdr_img, self.train_crop)
        else:
            hdr_img = ensure_even_hw(hdr_img, how="center_crop")
            sdr_img = ensure_even_hw(sdr_img, how="center_crop")

        # 5) 打包 tensor
        hdr_t = to_chw_tensor(hdr_img)  # [3,H,W]
        sdr_t = to_chw_tensor(sdr_img)

        H, W = hdr_t.shape[-2:]
        sample = {
            "sdr": sdr_t,               # FloatTensor[3,H,W], [0,1]
            "hdr": hdr_t,               # FloatTensor[3,H,W], [0,1]
            "meta": {
                "dataset": "HdM_HDR_2023_LogC3_AWG3",
                "is_video": False,
                "path_sdr": f"{path} | sdr_from={self.sdr_from} | ref={lut_meta}",
                "path_hdr": path,
                "size": (int(H), int(W)),
                "space_in": self.space_in,
                "space_out": self.space_out,
            }
        }
        return sample


def build_dataloader(cfg: Dict[str, Any], split: str, mode: str) -> torch.utils.data.DataLoader:
    """
    唯一对外函数：根据 cfg/split/mode 构建 PyTorch DataLoader
    - 仅支持 mode="img"；请求 "vid" 时抛 NotImplementedError
    """
    if mode not in ("img", "vid"):
        raise ValueError(f"Invalid mode: {mode}")

    if mode == "vid":
        raise NotImplementedError("Video mode not supported for this dataset.")

    batch_size = int(cfg.get("data", {}).get("batch_size", 1))
    num_workers = int(cfg.get("data", {}).get("num_workers", 4))
    pin_memory = bool(cfg.get("data", {}).get("pin_memory", True))
    drop_last = bool(cfg.get("data", {}).get("drop_last", split=="train"))

    dataset = ImgPairFromLogC3(cfg, split=split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split=="train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader
