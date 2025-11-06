import torch
import torch.nn.functional as F

DEFAULT_EPS = 1e-8

def _to_01(x: torch.Tensor) -> torch.Tensor:
    # Map to [0,1] when input looks like [-1,1]; otherwise clamp to [0,1]
    xmin, xmax = float(x.min().item()), float(x.max().item())
    if xmin >= -1.0 - 1e-6 and xmax <= 1.0 + 1e-6 and (xmin < 0.0 or xmax > 1.0):
        x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0)

@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor):
    """
    Returns: {"psnr": float}, batch-averaged over N.
    """
    dtype = torch.float32
    pred = _to_01(pred).to(dtype=dtype)
    target = _to_01(target).to(dtype=dtype)
    mse = F.mse_loss(pred, target, reduction='none')
    mse = mse.flatten(1).mean(dim=1)  # [N]
    psnr = 10.0 * torch.log10(1.0 / (mse + DEFAULT_EPS))  # data_range=1
    return {"psnr": float(psnr.mean().item())}
