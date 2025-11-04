# SDR2log

Training pipeline for reconstructing Log/ACEScct HDR imagery from SDR-domain inputs. The project targets both **image** and **video** stages; current work focuses on reproducing the image baselines that act as stepping stones toward the full pipeline.

## Project Goals
- Convert SDR (rendered from LogC3 captures) into ACEScct-space HDR representations.
- Establish dependable baselines before introducing novel HDR expansion models.
- Maintain modular data I/O, color management, and evaluation utilities that can be reused in the upcoming video stage.

## Baseline Roadmap (Image Stage)
1. **Non-learning pipelines**
   - Direct inverse mapping + ACES conversion.
   - Variant with highlight expansion to better preserve bright regions.
2. **General purpose regressors**
   - U-Net regressing ACEScct outputs directly.
   - U-Net regressing linear ACES followed by a CCT transfer.
3. **HDR expansion model adaptation (in progress)**
   - Existing HDR expander refit to output ACEScct; losses rewritten to supervise in the new domain.

The repository currently concentrates on item 3 above; earlier baselines are tracked for comparison once training scripts are stabilized.

## Repository Layout
- `baselines/` — baseline implementations; `img2img/diff-hdrtv` hosts the adapted diffusion HDR expansion model.
- `libs/` — shared utilities:
  - `color/` — TIFF loading, LUT application, color-space helpers.
  - `dataloaders/` — dataset construction (HdM HDR 2023 image pairs).
  - `losses/`, `metrics/` — pluggable loss and evaluation modules.
- `environment.yml` — reproducible Conda environment (CUDA 12.4 toolchain).

See `doc.md` for a consolidated guide to the per-module documentation.

## Getting Started
```bash
conda env create -f environment.yml
conda activate sdr2log
```

Prepare dataset assets (LogC3 TIFFs, LUTs) following the instructions in `libs/dataloaders/dataloader.md`, then launch training:
```bash
cd baselines/img2img/diff-hdrtv
python train.py --config config.yaml
```

The training script will emit experiment artifacts under `ckpt/` following the structure outlined in `baselines/img2img/ckpt.md`.

## Current Status & Next Steps
- ✅ Diff-HDRTV image baseline scaffolded with modular losses/metrics.
- ✅ Dataloader wired to HdM HDR 2023 dataset with color/LUT utilities.
- 🚧 Finalizing the HDR expansion model modifications (ACEScct outputs & losses).
- 🔜 Extend pipeline to video stage and implement color-space conversions beyond identity.

Contributions and experiments should document assumptions (dataset splits, LUTs, color-space tags) to stay reproducible across the upcoming video-stage development.

## TODO
- Complete ACEScct loss retargeting and diffusion prior tuning for the HDR expansion baseline.
- Re-benchmark non-learning and U-Net baselines against the new dataloader pipeline.
- Implement full color-space transforms in `convert_space` (LogC3↔ACES, HLG/PQ paths).
- Stand up evaluation utilities in `baselines/img2img/diff-hdrtv/eval.py`.
- Start drafting video-stage dataloaders once image baselines are locked.
