# -*- coding: utf-8 -*-
"""
Train script for Diff-HDRTV in **ACEScct + image** mode, rebuilt to satisfy ckpt.md,
and refactored to use external metrics modules following the unified interface:

Each metrics file (e.g., libs/metrics/psnr.py) must implement:
    def compute_metrics(pred, target) -> Dict[str, float or 0-dim torch.Tensor]

This script dynamically loads a list of metric modules from cfg['paths']['metrics'].
"""

import argparse
import importlib.util
import io
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import build_model

try:
    from torchvision.utils import save_image
except Exception:
    save_image = None  # fallback later

BARK_PUSH_URL = os.environ.get("BARK_URL") or os.environ.get("BARK_ENDPOINT")

# -----------------------------
# Small helpers
# -----------------------------

def _load_yaml(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _dyn_import_from_path(py_path: str, symbol: Optional[str] = None):
    py_path = os.path.abspath(py_path)
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Python path not found: {py_path}")
    mod_name = Path(py_path).stem + f"_{hash(py_path) & 0xFFFF:X}"
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return getattr(module, symbol) if symbol else module


def _seed_everything(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _save_yaml(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def _gather_env(seed: int) -> dict:
    env = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
        "amp": True if torch.cuda.is_available() else False,
        "python": sys.version.split(" ")[0],
        "seed": seed,
        "time": time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    try:
        import diffusers  # type: ignore
        env["diffusers"] = getattr(diffusers, "__version__", None)
    except Exception:
        env["diffusers"] = None
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        env["git_commit"] = commit
    except Exception:
        env["git_commit"] = None
    return env


def _notify_bark(title: str, body: str):
    if not BARK_PUSH_URL:
        return
    try:
        base = BARK_PUSH_URL.strip()
        if not base:
            return
        if "://" not in base:
            base = f"https://api.day.app/{base.lstrip('/')}"
        parsed = urllib.parse.urlparse(base)
        if parsed.netloc:
            segments = [seg for seg in parsed.path.split('/') if seg]
            if segments:
                token = segments[0]
                base = urllib.parse.urlunparse(
                    parsed._replace(path='/' + token, params='', query='', fragment='')
                )
        base = base.rstrip("/")
        encoded_title = urllib.parse.quote(title)
        encoded_body = urllib.parse.quote(body)
        url = f"{base}/{encoded_title}/{encoded_body}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            resp.read()
    except Exception as exc:  # pragma: no cover
        print(f"[Warn] Bark notification failed: {exc}")


# -----------------------------
# Loss aggregator (compute_loss API)
# -----------------------------
class LossBundle:
    def __init__(self, entries: Optional[List[dict]]):
        self.items = []
        if not entries:
            self.items = [("l1", 1.0, lambda p, t: F.l1_loss(p, t))]
            return
        for i, ent in enumerate(entries):
            path = ent.get("path")
            weight = float(ent.get("weight", 1.0))
            if path is None:
                raise KeyError(f"losses[{i}]: missing 'path'")
            mod = _dyn_import_from_path(path)
            if not hasattr(mod, 'compute_loss'):
                raise AttributeError(f"{path} must define compute_loss(pred, target)")
            self.items.append((Path(path).stem, weight, mod.compute_loss))

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        total = torch.zeros((), device=pred.device, dtype=pred.dtype)
        parts = {}
        for name, w, fn in self.items:
            val = fn(pred, target)
            if val.dim() != 0:
                val = val.mean()
            parts[f"loss_{name}"] = val.detach()
            total = total + w * val
        parts['loss_total'] = total
        return parts


# -----------------------------
# Metrics aggregator (compute_metrics API)
# -----------------------------
class MetricsBundle:
    """
    Load a list of metrics files, each implementing:
        compute_metrics(pred, target) -> Dict[str, float or 0-dim Tensor]
    and merge their outputs. Returns Dict[str, float].
    """
    def __init__(self, metric_paths: Optional[List[str]]):
        self.metric_fns = []
        if metric_paths:
            paths = metric_paths
        else:
            # Fallback defaults
            paths = [
                os.path.join("libs", "metrics", "psnr.py"),
                os.path.join("libs", "metrics", "ssim.py"),
                os.path.join("libs", "metrics", "mae.py"),
            ]
        for p in paths:
            mod = _dyn_import_from_path(p)
            if not hasattr(mod, "compute_metrics"):
                raise AttributeError(f"{p} must define compute_metrics(pred, target)")
            self.metric_fns.append((Path(p).stem, mod.compute_metrics))

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for _, fn in self.metric_fns:
            out = fn(pred, target)
            if not isinstance(out, dict):
                raise TypeError("compute_metrics must return a dict")
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    if v.dim() != 0:
                        raise ValueError(f"Metric '{k}' must be a scalar tensor")
                    merged[k] = float(v.item())
                else:
                    merged[k] = float(v)
        return merged


# -----------------------------
# Validation & sampling
# -----------------------------

@torch.no_grad()
def run_validation(model: nn.Module,
                   loader: Optional[DataLoader],
                   device: torch.device,
                   amp: bool,
                   metrics_bundle: MetricsBundle) -> Dict[str, float]:
    if loader is None:
        return {}
    model.eval()

    # online aggregation（加权平均：按样本数汇总）
    total = {}
    count = 0

    autocast = torch.cuda.amp.autocast if amp else torch.cpu.amp.autocast
    for batch in tqdm(loader, desc="val", leave=False):
        if 'sdr' not in batch:
            raise KeyError("Expect 'sdr' and 'hdr' keys in img mode.")
        x = batch['sdr'].to(device, non_blocking=True)
        y = batch['hdr'].to(device, non_blocking=True)
        with autocast():
            out = model(x)
            pred = torch.clamp(out['y'], 0.0, 1.0)

        # metrics_bundle 内部已 batch-mean；这里按批大小做样本加权平均
        m = metrics_bundle(pred, y)
        bsz = x.shape[0]
        for k, v in m.items():
            total[k] = total.get(k, 0.0) + v * bsz
        count += bsz

    model.train()
    if count == 0:
        return {}
    avg = {k: v / count for k, v in total.items()}
    # 附带样本数
    avg["n_images"] = float(count)
    return avg


def dump_samples(model: nn.Module, val_loader: DataLoader, out_samples: Path, epoch: int, k: int, device: torch.device, amp: bool):
    if save_image is None:
        print("[Warn] torchvision not available; skip saving samples.")
        return
    _ensure_dir(out_samples)
    saved = 0
    epoch_dir = _ensure_dir(out_samples / f"epoch_{epoch:04d}")
    autocast = torch.cuda.amp.autocast if amp else torch.cpu.amp.autocast
    with torch.no_grad():
        for batch in val_loader:
            if 'sdr' not in batch:
                continue
            x = batch['sdr'].to(device, non_blocking=True)
            y = batch['hdr'].to(device, non_blocking=True)
            with autocast():
                out = model(x)
                pred = torch.clamp(out['y'], 0.0, 1.0)
            diff = torch.clamp((pred - y).abs(), 0.0, 1.0)
            B = x.shape[0]
            for i in range(B):
                idx = saved + 1
                save_image(x[i], epoch_dir / f"img_{idx:03d}_input.png")
                save_image(pred[i], epoch_dir / f"img_{idx:03d}_pred.png")
                save_image(y[i], epoch_dir / f"img_{idx:03d}_gt.png")
                save_image(diff[i], epoch_dir / f"img_{idx:03d}_diff.png")
                saved += 1
                if saved >= k:
                    return


# -----------------------------
# Main training
# -----------------------------

def train(cfg: dict):
    # ---- mode checks ----
    if cfg.get('model', {}).get('mode', 'aces') != 'aces':
        raise NotImplementedError("This script supports ACEScct mode only (model.mode='aces').")
    if cfg.get('training', {}).get('task_mode', 'img') != 'img':
        raise NotImplementedError("This script supports image mode only (training.task_mode='img').")

    # ---- seed/device ----
    seed = int(cfg.get('seed', 42))
    _seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- dataloaders ----
    dl_path = cfg.get('paths', {}).get('dataloader')
    if dl_path is None:
        raise KeyError("config.paths.dataloader is required")
    build_dataloader = _dyn_import_from_path(dl_path, 'build_dataloader')
    train_loader: DataLoader = build_dataloader(cfg, split='train', mode='img')
    try:
        val_loader: Optional[DataLoader] = build_dataloader(cfg, split='val', mode='img')
    except Exception:
        val_loader = None

    # ---- model ----
    diff_cfg = cfg.get('model', {}).get('diffusion', {})
    model = build_model(
        mode='aces',
        low_res=tuple(diff_cfg.get('low_res', [128, 256])),
        num_inference_steps=int(diff_cfg.get('num_inference_steps', 50)),
        noise_std=float(diff_cfg.get('noise_std', 0.1)),
    ).to(device)

    # ---- losses ----
    criterion = LossBundle(cfg.get('paths', {}).get('losses', None))

    # ---- metrics ----
    metrics_bundle = MetricsBundle(cfg.get('paths', {}).get('metrics', None))
    main_metric = str(cfg.get('training', {}).get('main_metric', 'psnr')).lower()

    # ---- optimizer & sched ----
    tr = cfg.get('training', {})
    lr = float(tr.get('lr', 2e-4))
    wd = float(tr.get('weight_decay', 0.0))
    betas = tuple(tr.get('betas', [0.9, 0.999]))
    opt_name = str(tr.get('optimizer', 'adamw')).lower()
    if opt_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    sched_cfg = tr.get('scheduler', {"name": "cosine", "t_max": tr.get('epochs', 1)})
    sched_name = str(sched_cfg.get('name', 'cosine')).lower()
    if sched_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(sched_cfg.get('t_max', tr.get('epochs', 1))))
        )
    elif sched_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(sched_cfg.get('step_size', 10)),
            gamma=float(sched_cfg.get('gamma', 0.5))
        )
    else:
        scheduler = None

    # ---- AMP / grad ----
    amp = bool(tr.get('amp', True)) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    grad_accum = int(tr.get('grad_accum_steps', 1))
    clip_grad = float(tr.get('clip_grad_norm', 0.0) or 0.0)

    # ---- ckpt root & exp dir ----
    ckpt_root = Path(tr.get('ckpt_root', 'ckpt'))
    date_str = time.strftime('%Y%m%d')
    exp_name = str(tr.get('exp_name', 'exp'))
    exp_dir = ckpt_root / f"{date_str}-{exp_name}"

    paths = {
        'config_dir': _ensure_dir(exp_dir / 'config'),
        'meta_dir': _ensure_dir(exp_dir / 'meta'),
        'ckpt_dir': _ensure_dir(exp_dir / 'checkpoints'),
        'logs_dir': _ensure_dir(exp_dir / 'logs'),
        'eval_dir': _ensure_dir(exp_dir / 'eval'),
        'samples_dir': _ensure_dir(exp_dir / 'samples'),
    }

    # ---- dump split configs & env ----
    _save_yaml(paths['config_dir'] / 'data.yaml', cfg.get('data', {}))
    _save_yaml(paths['config_dir'] / 'model.yaml', cfg.get('model', {}))
    optim_dump = {k: v for k, v in cfg.get('training', {}).items() if k not in ('ckpt_root', 'exp_name')}
    _save_yaml(paths['config_dir'] / 'optim.yaml', optim_dump)
    with open(paths['meta_dir'] / 'env.json', 'w', encoding='utf-8') as f:
        json.dump(_gather_env(seed), f, ensure_ascii=False, indent=2)

    # ---- resume (optional) ----
    resume_path = tr.get('resume', None)
    start_epoch = 0
    global_step = 0
    best_val = -float('inf')
    if resume_path:
        ckpt = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt and ckpt['scaler'] is not None:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt.get('epoch', 0)
        global_step = ckpt.get('global_step', 0)
        best_val = ckpt.get('best_val', best_val)
        print(f"Resumed from {resume_path} @ epoch {start_epoch}, step {global_step}")

    # ---- training loop ----
    epochs = int(tr.get('epochs', 1))
    log_train_path = paths['logs_dir'] / 'scalars.train.jsonl'
    log_val_path = paths['logs_dir'] / 'scalars.val.jsonl'
    sample_every_epochs = int(tr.get('sample_every_epochs', 5))
    sample_k = int(tr.get('sample_k', 8))

    model.train()
    autocast = torch.cuda.amp.autocast if amp else torch.cpu.amp.autocast

    try:
        for epoch in range(start_epoch, epochs):
            t0_epoch = time.time()
            optimizer.zero_grad(set_to_none=True)
            last_train_loss: Optional[float] = None

            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"epoch {epoch}")
            for it, batch in pbar:
                t0 = time.time()
                if 'sdr' not in batch:
                    raise KeyError("Expect 'sdr' and 'hdr' keys in img mode.")
                x = batch['sdr'].to(device, non_blocking=True)
                y = batch['hdr'].to(device, non_blocking=True)

                with autocast():
                    out = model(x)
                    pred = out['y']
                    loss_dict = criterion(pred, y)
                    loss = loss_dict['loss_total'] / grad_accum
                    last_train_loss = float(loss_dict['loss_total'].item())

                scaler.scale(loss).backward()

                if (it + 1) % grad_accum == 0:
                    if clip_grad > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1
                iter_s = time.time() - t0

                if (it + 1) % max(1, int(tr.get('log_every', 50))) == 0:
                    lr_now = optimizer.param_groups[0]['lr']
                    log_obj = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss_total": float(loss_dict['loss_total'].item()),
                        **{k: float(v.item()) for k, v in loss_dict.items() if k != 'loss_total'},
                        "lr": float(lr_now),
                        "iter_s": float(iter_s),
                    }
                    _write_jsonl(log_train_path, log_obj)
                    pbar.set_postfix({"loss": f"{log_obj['loss_total']:.4f}", "lr": f"{lr_now:.2e}"})

            # step scheduler per-epoch if defined that way
            if scheduler is not None:
                scheduler.step()

            # ---- validation ----
            val_stats = {}
            if val_loader is not None:
                val_stats = run_validation(model, val_loader, device, amp, metrics_bundle)
                val_log = {"epoch": epoch, **val_stats}
                _write_jsonl(log_val_path, val_log)

                # save best by main_metric
                if main_metric not in val_stats:
                    print(f"[Warn] main_metric '{main_metric}' not in validation stats. "
                          f"Available: {list(val_stats.keys())}")
                else:
                    cur = float(val_stats[main_metric])
                    is_best = cur > best_val
                    if is_best:
                        best_val = cur
                        torch.save({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scaler': scaler.state_dict() if amp else None,
                            'epoch': epoch,
                            'global_step': global_step,
                            'best_val': best_val,
                            'main_metric': main_metric,
                            'cfg': cfg,
                        }, paths['ckpt_dir'] / f"best@{main_metric}.pt")

            # ---- samples ----
            if (val_loader is not None) and ((epoch + 1) % sample_every_epochs == 0):
                dump_samples(model, val_loader, paths['samples_dir'], epoch + 1, sample_k, device, amp)

            # ---- save last each epoch ----
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if amp else None,
                'epoch': epoch + 1,
                'global_step': global_step,
                'best_val': best_val,
                'main_metric': main_metric,
                'cfg': cfg,
            }, paths['ckpt_dir'] / 'last.pt')

            # ---- end-of-epoch report ----
            t1_epoch = time.time()
            best_str = f"{best_val:.3f}" if best_val != -float('inf') else "N/A"
            print(f"Epoch {epoch+1}/{epochs} done in {t1_epoch - t0_epoch:.1f}s. "
                  f"Best {main_metric}: {best_str}")

            # ---- Bark notification ----
            bark_interval = int(cfg.get("training", {}).get("bark_every_epochs", 10))
            if BARK_PUSH_URL and bark_interval > 0 and ((epoch + 1) % bark_interval == 0):
                body_parts = []
                if last_train_loss is not None:
                    body_parts.append(f"train_loss={last_train_loss:.4f}")
                if val_stats:
                    metric_items = []
                    for k, v in val_stats.items():
                        if isinstance(v, (int, float)):
                            if isinstance(v, float):
                                metric_items.append(f"{k}={v:.4f}")
                            else:
                                metric_items.append(f"{k}={v}")
                    if metric_items:
                        body_parts.append("val " + ", ".join(metric_items))
                if best_val != -float('inf'):
                    body_parts.append(f"best_{main_metric}={best_val:.4f}")
                elif body_parts == []:
                    body_parts.append("No metrics available")
                _notify_bark(
                    title=f"Epoch {epoch+1}/{epochs}",
                    body=" | ".join(body_parts)
                )

        # ---- final eval summary ----
        if val_loader is not None:
            stats = run_validation(model, val_loader, device, amp, metrics_bundle)
            with open(paths['eval_dir'] / 'summary.json', 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

    except KeyboardInterrupt:
        print("Interrupted. Saving last.pt and exiting.")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if amp else None,
            'epoch': epoch if 'epoch' in locals() else 0,
            'global_step': global_step,
            'best_val': best_val,
            'main_metric': main_metric,
            'cfg': cfg,
        }, paths['ckpt_dir'] / 'last.pt')
        raise


# -----------------------------
# Entry
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        here = Path(__file__).resolve().parent
        alt = here / 'config.yaml'
        if alt.exists():
            cfg_path = alt
        else:
            raise FileNotFoundError(f"config.yaml not found at {args.config} or {alt}")

    cfg = _load_yaml(cfg_path)

    # Surface intended mode
    if cfg.get('model', {}).get('mode', 'aces') != 'aces':
        print("[Warning] model.mode should be 'aces' for this script.")
    if cfg.get('training', {}).get('task_mode', 'img') != 'img':
        print("[Warning] training.task_mode should be 'img' for this script.")

    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    train(cfg)


if __name__ == '__main__':
    main()
