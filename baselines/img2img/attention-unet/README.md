# Attention U-Net Baseline

A lightweight single-stage baseline that maps SDR inputs (ACES -> Rec.709 -> ACES, produced by the shared HdM HDR dataloader) to HDR ACEScct targets using an Attention U-Net.

## Files
- `model.py`: Attention U-Net with additive attention gates and configurable channel depths.
- `train.py`: Training loop that reuses the shared dataloader, loss, and metric modules.
- `config.yaml`: Example configuration for HdM HDR 2023 (adjust the absolute paths to match your workspace if needed).

## Training
```bash
cd baselines/img2img/attention-unet
python train.py --config config.yaml --device auto
```
`--device auto` selects CUDA when available. Checkpoints and logs are written under `training.ckpt_root/exp_name`.
