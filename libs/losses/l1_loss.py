import torch
import torch.nn.functional as F

def compute_loss(pred, target):
    """
    计算预测结果与目标之间的 L1 损失（Mean Absolute Error）。

    Args:
        pred (torch.Tensor): 模型输出的预测图像，形状为 (N, C, H, W)。
        target (torch.Tensor): 对应的目标图像，形状为 (N, C, H, W)。

    Returns:
        torch.Tensor: 标量张量，表示该批次的平均 L1 损失值。
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    
    loss = F.l1_loss(pred, target, reduction='mean')
    return loss
