#!/usr/bin/env python3
"""
Utility script to train Diff-HDRTV in three stages sequentially.
Stage-1: HDRNet only
Stage-2: Diffusion prior only (exports prior_only.pt)
Stage-3: C/L/Refiner with frozen prior (auto-loads latest prior checkpoint)
"""

import subprocess
import sys
import yaml
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
TRAIN_SCRIPT = HERE / "train.py"


def _run_training(config_path: Path):
    print(f"[Run] python train.py --config {config_path}")
    subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)],
        check=True,
        cwd=HERE,
    )


def _latest_prior_checkpoint(ckpt_root: Path, exp_name_suffix: str) -> Path:
    pattern = f"*{exp_name_suffix}"
    candidates = sorted(
        ckpt_root.glob(pattern),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No experiment dir matching '{pattern}' under {ckpt_root}")
    for exp_dir in candidates:
        prior_path = exp_dir / "checkpoints" / "prior_only.pt"
        if prior_path.exists():
            return prior_path
    raise FileNotFoundError(f"prior_only.pt not found under any '{pattern}' experiment in {ckpt_root}")


def _prepare_stage3_config(base_cfg: Path, prior_path: Path) -> Path:
    cfg = yaml.safe_load(base_cfg.read_text())
    cfg.setdefault("training", {})
    cfg["training"]["load_prior_path"] = str(prior_path)
    temp_cfg = base_cfg.with_name(base_cfg.stem + "_auto.yaml")
    with open(temp_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return temp_cfg


def main():
    cfg_stage1 = HERE / "config_stage1.yaml"
    cfg_stage2 = HERE / "config_stage2.yaml"
    cfg_stage3 = HERE / "config_stage3.yaml"

    for cfg in (cfg_stage1, cfg_stage2, cfg_stage3):
        if not cfg.exists():
            raise FileNotFoundError(f"Missing config file: {cfg}")

    _run_training(cfg_stage1)
    _run_training(cfg_stage2)

    stage2_prior = _latest_prior_checkpoint(
        ckpt_root=Path(yaml.safe_load(cfg_stage2.read_text())["training"]["ckpt_root"]),
        exp_name_suffix="diffh_stage2",
    )
    print(f"[Info] Using prior checkpoint: {stage2_prior}")

    stage3_temp_cfg: Optional[Path] = None
    try:
        stage3_temp_cfg = _prepare_stage3_config(cfg_stage3, stage2_prior)
        _run_training(stage3_temp_cfg)
    finally:
        if stage3_temp_cfg and stage3_temp_cfg.exists():
            stage3_temp_cfg.unlink()


if __name__ == "__main__":
    main()

