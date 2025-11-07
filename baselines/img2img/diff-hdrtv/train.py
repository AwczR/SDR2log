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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def _seed_everything(seed: int, deterministic: bool = True):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


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


def _stage_train_targets(model: nn.Module, stage: int, detach_prior_in_stage3: bool) -> List[str]:
    if stage == 1:
        return ["hdrnet"]
    if stage == 2:
        return ["prior"]
    if stage == 3:
        if getattr(model, "mode", "aces") == "aces":
            targets = ["decomp", "cnet", "lnet", "refiner_aces3"]
        else:
            targets = ["xynet", "ynet", "refiner_xyY"]
        if not detach_prior_in_stage3:
            targets.append("prior")
        return targets
    raise ValueError(f"Unsupported training.stage={stage}. Expected 1, 2, or 3.")


def _apply_stage_freeze(model: nn.Module, stage: int, detach_prior_in_stage3: bool) -> List[str]:
    targets = set(_stage_train_targets(model, stage, detach_prior_in_stage3))
    known_attrs = [
        "hdrnet", "prior",
        "decomp", "cnet", "lnet", "refiner_aces3",
        "xynet", "ynet", "refiner_xyY",
    ]
    enabled = []
    for name in known_attrs:
        module = getattr(model, name, None)
        if module is None:
            continue
        requires = name in targets
        for p in module.parameters():
            p.requires_grad = requires
        if requires:
            enabled.append(name)
    return enabled


def _should_detach_prior(stage: int, detach_prior_in_stage3: bool) -> bool:
    if stage == 2:
        return False
    if stage == 3:
        return detach_prior_in_stage3
    return True  # Stage-1 默认只训练 HDRNet，避免反传进 prior


def _aces_ortho_regularizer(model: nn.Module) -> torch.Tensor:
    if not hasattr(model, "decomp"):
        raise AttributeError("ACES mode required for orthogonal regularization.")
    weight = model.decomp.proj.weight.view(model.decomp.proj.weight.shape[0], -1)
    ident = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
    diff = weight @ weight.t() - ident
    return torch.sum(diff * diff)


def _save_prior_only(prior: nn.Module, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'prior': prior.state_dict()}, out_path)


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
                   metrics_bundle: MetricsBundle,
                   detach_prior: bool,
                   use_prior: bool,
                   stage: int) -> Dict[str, float]:
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
            out = model(x, detach_prior=detach_prior, use_prior=use_prior)
            pred = out['hdr_ref'] if stage == 1 else out['y']
            pred_vis = pred.clamp(0.0, 1.0)

        # metrics_bundle 内部已 batch-mean；这里按批大小做样本加权平均
        m = metrics_bundle(pred_vis, y)
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


def dump_samples(model: nn.Module, val_loader: DataLoader, out_samples: Path, epoch: int, k: int,
                 device: torch.device, amp: bool, detach_prior: bool, use_prior: bool, stage: int):
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
                out = model(x, detach_prior=detach_prior, use_prior=use_prior)
                pred = out['hdr_ref'] if stage == 1 else out['y']
                pred_vis = pred.clamp(0.0, 1.0)
            diff = torch.clamp((pred_vis - y).abs(), 0.0, 1.0)
            B = x.shape[0]
            for i in range(B):
                idx = saved + 1
                save_image(x[i], epoch_dir / f"img_{idx:03d}_input.png")
                save_image(pred_vis[i], epoch_dir / f"img_{idx:03d}_pred.png")
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

    tr = cfg.get('training', {})
    stage = int(tr.get('stage', 3))
    detach_prior_in_stage3 = bool(tr.get('detach_prior_in_stage3', True))
    use_prior_in_stage3 = bool(tr.get('use_prior_in_stage3', False))
    deterministic = bool(tr.get('deterministic', True))

    # ---- seed/device ----
    seed = int(cfg.get('seed', 42))
    _seed_everything(seed, deterministic=deterministic)
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
        input_range=str(diff_cfg.get('input_range', 'zero_one')),
        prediction_type=str(diff_cfg.get('prediction_type', 'epsilon')),
        resize_mode=str(diff_cfg.get('resize_mode', 'bicubic')),
    ).to(device)

    load_prior_path = tr.get('load_prior_path')
    if load_prior_path:
        prior_path = Path(load_prior_path).expanduser()
        if not prior_path.exists():
            raise FileNotFoundError(f"training.load_prior_path not found: {prior_path}")
        ckpt = torch.load(prior_path, map_location='cpu')
        if 'prior' not in ckpt:
            raise KeyError(f"{prior_path} must contain a 'prior' key.")
        model.prior.load_state_dict(ckpt['prior'])
        print(f"[Info] Loaded prior weights from {prior_path}")

    trainable_module_names = _apply_stage_freeze(model, stage, detach_prior_in_stage3)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(f"No trainable parameters found for stage={stage}.")
    detach_prior_flag = _should_detach_prior(stage, detach_prior_in_stage3)
    use_prior_flag = (stage == 2) or (stage == 3 and use_prior_in_stage3)

    print(
        "[Contract] diffusion.input_range={inp}, prediction_type={pred}, noise_std={noise}, resize_mode={resize}".format(
            inp=diff_cfg.get('input_range', 'zero_one'),
            pred=diff_cfg.get('prediction_type', 'epsilon'),
            noise=diff_cfg.get('noise_std', 0.1),
            resize=diff_cfg.get('resize_mode', 'bicubic'),
        )
    )
    print("[Contract] training.stage={stage}, detach_prior_in_stage3={det}, use_prior_in_stage3={usep}, detach_prior_flag={flag}, trainable_modules={mods}".format(
        stage=stage,
        det=detach_prior_in_stage3,
        usep=use_prior_in_stage3,
        flag=detach_prior_flag,
        mods=", ".join(trainable_module_names) or "None",
    ))

    # ---- losses ----
    criterion = LossBundle(cfg.get('paths', {}).get('losses', None))

    # ---- metrics ----
    metrics_bundle = MetricsBundle(cfg.get('paths', {}).get('metrics', None))
    main_metric = str(cfg.get('training', {}).get('main_metric', 'psnr')).lower()

    # ---- optimizer & sched ----
    lr = float(tr.get('lr', 2e-4))
    lr_prior = float(tr.get('lr_prior', lr))
    opt_lr = lr_prior if stage == 2 else lr
    wd = float(tr.get('weight_decay', 0.0))
    betas = tuple(tr.get('betas', [0.9, 0.999]))
    opt_name = str(tr.get('optimizer', 'adamw')).lower()
    if opt_name == 'adamw':
        optimizer = torch.optim.AdamW(trainable_params, lr=opt_lr, betas=betas, weight_decay=wd)
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(trainable_params, lr=opt_lr, betas=betas, weight_decay=wd)
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
    lambda_ortho = float(tr.get('lambda_ortho', 0.0) or 0.0)

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
                    out = model(x, detach_prior=detach_prior_flag, use_prior=use_prior_flag)
                    pred = out['hdr_ref'] if stage == 1 else out['y']
                    loss_dict = criterion(pred, y)
                    main_loss = loss_dict['loss_total']
                    total_loss = main_loss
                    if lambda_ortho > 0 and hasattr(model, "decomp"):
                        ortho_loss = _aces_ortho_regularizer(model)
                        loss_dict['loss_ortho'] = ortho_loss.detach()
                        total_loss = total_loss + lambda_ortho * ortho_loss
                    loss = total_loss / grad_accum
                loss_dict['loss_main'] = main_loss.detach()
                loss_dict['loss_total'] = total_loss.detach()
                last_train_loss = float(loss_dict['loss_total'].item())

                scaler.scale(loss).backward()

                if (it + 1) % grad_accum == 0:
                    if clip_grad > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_grad)
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
                val_stats = run_validation(model, val_loader, device, amp, metrics_bundle, detach_prior_flag, use_prior_flag, stage)
                val_log = {"epoch": epoch, **val_stats}
                _write_jsonl(log_val_path, val_log)

                # save best by main_metric
                if main_metric not in val_stats:
                    raise KeyError(f"main_metric '{main_metric}' missing from validation stats. Available keys: {list(val_stats.keys())}")
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
                    if stage == 2:
                        _save_prior_only(model.prior, paths['ckpt_dir'] / 'prior_only.pt')

            # ---- samples ----
            if (val_loader is not None) and ((epoch + 1) % sample_every_epochs == 0):
                dump_samples(model, val_loader, paths['samples_dir'], epoch + 1, sample_k, device, amp, detach_prior_flag, use_prior_flag, stage)

            # ---- save last each epoch ----
            last_payload = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if amp else None,
                'epoch': epoch + 1,
                'global_step': global_step,
                'best_val': best_val,
                'main_metric': main_metric,
                'cfg': cfg,
            }
            torch.save(last_payload, paths['ckpt_dir'] / 'last.pt')
            if stage == 2:
                _save_prior_only(model.prior, paths['ckpt_dir'] / 'prior_only.pt')

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
            stats = run_validation(model, val_loader, device, amp, metrics_bundle, detach_prior_flag, use_prior_flag, stage)
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
        if stage == 2:
            _save_prior_only(model.prior, paths['ckpt_dir'] / 'prior_only.pt')
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
