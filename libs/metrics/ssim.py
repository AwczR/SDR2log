import torch
import torch.nn.functional as F

def _to_01(x: torch.Tensor) -> torch.Tensor:
    xmin, xmax = float(x.min().item()), float(x.max().item())
    if xmin >= -1.0 - 1e-6 and xmax <= 1.0 + 1e-6 and (xmin < 0.0 or xmax > 1.0):
        x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0)

def _gaussian_window(ch: int, k: int = 11, sigma: float = 1.5, device=None, dtype=None):
    coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = (g / g.sum()).unsqueeze(0)
    win2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    win = win2d.repeat(ch, 1, 1, 1)
    return win

@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor):
    """
    Returns: {"ssim": float}, batch-averaged over N.
    (Simple channel-wise SSIM averaged over channels on [0,1])
    """
    pred = _to_01(pred)
    target = _to_01(target)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    N, C, H, W = pred.shape
    device = pred.device
    dtype = torch.float32
    pred = pred.to(dtype=dtype)
    target = target.to(dtype=dtype)
    win = _gaussian_window(C, 11, 1.5, device=device, dtype=dtype)
    padding = 11 // 2

    mu1 = F.conv2d(pred, win, groups=C, padding=padding)
    mu2 = F.conv2d(target, win, groups=C, padding=padding)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, win, groups=C, padding=padding) - mu1_sq
    sigma2_sq = F.conv2d(target * target, win, groups=C, padding=padding) - mu2_sq
    sigma12 = F.conv2d(pred * target, win, groups=C, padding=padding) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    ssim = ssim_map.flatten(1).mean(dim=1)  # [N]
    return {"ssim": float(ssim.mean().item())}
