import torch
import torch.nn.functional as F

def _to_01(x: torch.Tensor) -> torch.Tensor:
    xmin, xmax = float(x.min().item()), float(x.max().item())
    if xmin >= -1.0 - 1e-6 and xmax <= 1.0 + 1e-6 and (xmin < 0.0 or xmax > 1.0):
        x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0)

@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor):
    """
    Returns: {"mae": float}, batch-averaged over N.
    """
    dtype = torch.float32
    pred = _to_01(pred).to(dtype=dtype)
    target = _to_01(target).to(dtype=dtype)
    mae = torch.nn.functional.l1_loss(pred, target, reduction='none')
    mae = mae.flatten(1).mean(dim=1)  # [N]
    return {"mae": float(mae.mean().item())}
