### 接口文档：`loss.py`

#### 函数接口
```python
def compute_loss(pred, target):
    """
    计算预测结果与目标之间的损失。

    Args:
        pred (torch.Tensor): 模型输出的预测图像，形状为 (N, C, H, W)。
        target (torch.Tensor): 对应的目标图像，形状为 (N, C, H, W)。

    Returns:
        torch.Tensor: 标量张量，表示该批次的平均损失值。
    """
    ...
```

#### 设计说明
- **统一命名**：所有损失文件（如 `l1_loss.py`, `ssim_loss.py`, `perceptual_loss.py`）都必须实现 `compute_loss(pred, target)`。  
- **输入一致**：`pred` 与 `target` 均为相同 shape 的 `torch.Tensor`，值范围通常为 `[0, 1]` 或 `[-1, 1]`。  
- **输出为标量**：返回一个标量张量，方便在训练循环中直接 `loss.backward()`。  
- **独立可调用**：每个损失模块可单独 import 使用，例如：
  ```python
  from lib.losses.l1_loss import compute_loss
  loss = compute_loss(pred, target)
  ```


