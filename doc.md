# Documentation Map

This document gathers the module-level notes that currently live across the repository. Use it as a quick index when navigating the codebase.

## Baselines
- `baselines/img2img/ckpt.md` — experiment output layout for image-stage baselines, detailing `config/`, `meta/`, `checkpoints/`, `logs/`, `eval/`, and `samples/` folders.
- `baselines/img2img/diff-hdrtv/train.py` — training entry point for the adapted diffusion HDR expander. Key helpers inside the script:
  - Dynamic loaders (`_dyn_import_from_path`) for datasets, losses, and metrics defined in `config.yaml`.
  - `LossBundle` / `MetricsBundle` aggregators that expect each module to expose `compute_loss` / `compute_metrics`.
  - Checkpoint, logging, and sample dumping logic aligned with `ckpt.md`.

## Shared Libraries
- `libs/color/color.md` — authoritative guide for color & I/O helpers:
  - `libs/color/io.py` — TIFF loading and normalization (`read_tiff_as_float01`).
  - `libs/color/lut.py` — `.cube` LUT parsing (`load_cube_lut`) and application (`apply_3d_lut`).
  - `libs/color/ops.py` — geometry helpers (`ensure_even_hw`, `to_chw_tensor`) and the `convert_space` routing stub.
- `libs/dataloaders/dataloader.md` — contract for `build_dataloader(cfg, split, mode)`:
  - Describes expected sample dictionaries for `img` vs `vid`.
  - Current implementation: `libs/dataloaders/hdm-hdr-2023/dataloader.py`, which assembles LogC3↔SDR pairs using the color utilities.
- `libs/losses/loss.md` — interface requirements for individual loss modules (single `compute_loss(pred, target)` returning a scalar tensor). The default L1 implementation lives in `libs/losses/l1_loss.py`.
- `libs/metrics/metrics.md` — interface requirements for evaluation metrics (`compute_metrics` returning a dict of floats/scalar tensors). Implementations include PSNR, SSIM, and MAE under `libs/metrics/`.

## Environment & Reproducibility
- `environment.yml` — Conda spec targeting Python 3.10, PyTorch 2.4 (CUDA 12.4), and supporting libraries (`diffusers`, `colour-science`, etc.) required by the baselines.
- Training outputs (`ckpt/`) capture runtime metadata via `meta/env.json` and serialized configs under `config/`, ensuring reproducible runs.

Refer back to the module-level Markdown files for deeper explanations or implementation notes while extending the project.
