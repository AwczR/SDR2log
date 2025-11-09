# -*- coding: utf-8 -*-
"""Training script for the Attention U-Net baseline."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import build_model  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Attention U-Net baseline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda or cpu")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dyn_import(path: str):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    mod_name = f"loader_{hash(path) & 0xFFFF:X}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot import from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _seed_everything(seed: int, deterministic: bool = True):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def _build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    paths = cfg.get("paths", {})
    loader_path = paths.get("dataloader")
    if not loader_path:
        raise ValueError("paths.dataloader is required")
    module = _dyn_import(loader_path)
    if not hasattr(module, "build_dataloader"):
        raise AttributeError(f"`build_dataloader` not found in {loader_path}")
    task_mode = cfg.get("training", {}).get("task_mode", "img")
    train_loader = module.build_dataloader(cfg, split="train", mode=task_mode)
    val_loader = module.build_dataloader(cfg, split="val", mode=task_mode)
    return train_loader, val_loader


def _load_loss_modules(cfg: dict) -> List[Tuple[float, Callable]]:
    losses_cfg = cfg.get("paths", {}).get("losses", [])
    if not losses_cfg:
        raise ValueError("No loss modules specified under paths.losses")
    specs = []
    for entry in losses_cfg:
        weight = float(entry.get("weight", 1.0))
        module = _dyn_import(entry["path"])
        if not hasattr(module, "compute_loss"):
            raise AttributeError(f"compute_loss missing in {entry['path']}")
        specs.append((weight, module.compute_loss))
    return specs


def _load_metric_modules(cfg: dict) -> List[Callable]:
    metric_paths = cfg.get("paths", {}).get("metrics", [])
    modules = []
    for path in metric_paths:
        module = _dyn_import(path)
        if not hasattr(module, "compute_metrics"):
            raise AttributeError(f"compute_metrics missing in {path}")
        modules.append(module.compute_metrics)
    return modules


def _init_optimizer(model: nn.Module, train_cfg: dict) -> optim.Optimizer:
    lr = float(train_cfg.get("lr", 2e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    betas = train_cfg.get("betas", [0.9, 0.999])
    opt_name = train_cfg.get("optimizer", "adamw").lower()
    if opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, betas=tuple(betas), weight_decay=weight_decay)
    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, betas=tuple(betas), weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {opt_name}")


def _init_scheduler(optimizer: optim.Optimizer, train_cfg: dict):
    sched_cfg = train_cfg.get("scheduler")
    if not sched_cfg:
        return None
    name = sched_cfg.get("name", "cosine").lower()
    if name == "cosine":
        t_max = int(sched_cfg.get("t_max", train_cfg.get("epochs", 50)))
        eta_min = float(sched_cfg.get("eta_min", 1e-6))
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    if name == "step":
        step_size = int(sched_cfg.get("step_size", 10))
        gamma = float(sched_cfg.get("gamma", 0.5))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f"Unsupported scheduler: {name}")


def _prepare_dirs(train_cfg: dict) -> Dict[str, Path]:
    ckpt_root = Path(train_cfg.get("ckpt_root", "ckpt"))
    exp_name = train_cfg.get("exp_name", "attention_unet")
    exp_dir = ckpt_root / exp_name
    ckpt_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return {"exp_dir": exp_dir, "ckpt_dir": ckpt_dir, "log_dir": log_dir}


def _save_ckpt(path: Path, epoch: int, model: nn.Module, optimizer, scheduler, best_metric: float):
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
    }
    torch.save(payload, path)


def _load_ckpt(path: Path, model: nn.Module, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", float("-inf"))


def _write_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _run_epoch(model: nn.Module,
               loader: DataLoader,
               losses: List[Tuple[float, Callable]],
               optimizer,
               scaler: GradScaler,
               device: torch.device,
               grad_accum: int,
               max_norm: float,
               use_amp: bool) -> float:
    model.train()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)
    total_steps = len(loader)
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        sdr = batch["sdr"].to(device)
        hdr = batch["hdr"].to(device)
        with autocast(enabled=use_amp, device_type=device.type):
            preds = model(sdr)["pred"]
            loss = 0.0
            for weight, fn in losses:
                loss = loss + weight * fn(preds, hdr)
        loss = loss / grad_accum
        scaler.scale(loss).backward()
        should_step = ((step + 1) % grad_accum == 0) or ((step + 1) == total_steps)
        if should_step:
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running += loss.item() * grad_accum
    return running / max(total_steps, 1)


def _evaluate(model: nn.Module,
              loader: DataLoader,
              losses: List[Tuple[float, Callable]],
              metrics_fns: List[Callable],
              device: torch.device,
              use_amp: bool) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    metrics_sum: Dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            sdr = batch["sdr"].to(device)
            hdr = batch["hdr"].to(device)
            with autocast(enabled=use_amp, device_type=device.type):
                preds = model(sdr)["pred"]
                loss = 0.0
                for weight, fn in losses:
                    loss = loss + weight * fn(preds, hdr)
            total_loss += loss.item()
            for fn in metrics_fns:
                result = fn(preds, hdr)
                for k, v in result.items():
                    if isinstance(v, torch.Tensor):
                        if v.ndim > 0:
                            v = v.detach().mean()
                        else:
                            v = v.detach()
                        v = v.to("cpu").item()
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + float(v)
            count += 1
    metrics_avg = {k: v / max(count, 1) for k, v in metrics_sum.items()}
    return total_loss / max(count, 1), metrics_avg


def main():
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path)
    train_cfg = cfg.get("training", {})
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        requested = torch.device(args.device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            print("[Warn] CUDA not available, switching to CPU.")
            device = torch.device("cpu")
        else:
            device = requested
    _seed_everything(int(cfg.get("seed", 42)), bool(train_cfg.get("deterministic", True)))

    train_loader, val_loader = _build_dataloaders(cfg)
    model = build_model(**cfg.get("model", {})).to(device)
    optimizer = _init_optimizer(model, train_cfg)
    scheduler = _init_scheduler(optimizer, train_cfg)
    losses = _load_loss_modules(cfg)
    metrics_fns = _load_metric_modules(cfg)

    dirs = _prepare_dirs(train_cfg)
    snapshot_path = dirs["exp_dir"] / "config_snapshot.yaml"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    resume_path = train_cfg.get("resume")
    start_epoch = 0
    maximize_metric = bool(train_cfg.get("maximize_metric", True))
    best_metric = float("-inf") if maximize_metric else float("inf")
    if resume_path:
        start_epoch, best_metric = _load_ckpt(Path(resume_path), model, optimizer, scheduler)

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    max_norm = float(train_cfg.get("clip_grad_norm", 0.0))
    epochs = int(train_cfg.get("epochs", 50))
    main_metric = train_cfg.get("main_metric", "psnr")

    for epoch in range(start_epoch, epochs):
        train_loss = _run_epoch(model, train_loader, losses, optimizer, scaler, device, grad_accum, max_norm, use_amp)
        val_loss, metrics_avg = _evaluate(model, val_loader, losses, metrics_fns, device, use_amp)
        if scheduler:
            scheduler.step()

        metrics_avg = metrics_avg or {}
        if main_metric in metrics_avg:
            target_metric = metrics_avg[main_metric]
        else:
            target_metric = -val_loss if maximize_metric else val_loss
        is_better = target_metric > best_metric if maximize_metric else target_metric < best_metric
        if is_better:
            best_metric = target_metric
            _save_ckpt(dirs["ckpt_dir"] / "best.pth", epoch, model, optimizer, scheduler, best_metric)
        _save_ckpt(dirs["ckpt_dir"] / "last.pth", epoch, model, optimizer, scheduler, best_metric)

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "metrics": metrics_avg,
            "lr": optimizer.param_groups[0]["lr"],
        }
        _write_jsonl(dirs["log_dir"] / "metrics.jsonl", log_entry)
        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} {main_metric}={target_metric:.4f}")

    print("Training finished. Best metric:", best_metric)


if __name__ == "__main__":
    main()
