### 接口文档：`metrics.py`

#### 函数接口
```python
def compute_metrics(pred, target):
    """
    计算预测结果与目标之间的评价指标（无梯度参与）。

    Args:
        pred (torch.Tensor): 模型输出的预测图像，形状为 (N, C, H, W)。
        target (torch.Tensor): 对应的目标图像，形状为 (N, C, H, W)。

    Returns:
        Dict[str, Union[float, torch.Tensor]]:
            一个字典，键为指标名，值为该批次的平均指标值。
            - 若为 torch.Tensor，须为标量张量（shape == []）。
            - 建议返回 Python float 以便日志记录与序列化。
    """
    ...
```

#### 设计说明
- **统一命名**：所有指标文件（如 `psnr.py`, `ssim.py`, `lpips.py`）都必须实现同名接口 `compute_metrics(pred, target)`。
- **输入一致**：`pred` 与 `target` 均为相同 shape 的 `torch.Tensor`，典型值域为 `[0, 1]` 或 `[-1, 1]`（各指标内部自行处理值域与必要的 `clamp/normalize`）。
- **输出为字典**：返回 `Dict[str, float/ScalarTensor]`。键名使用小写加下划线（如 `"psnr"`, `"ssim"`, `"lpips"`），数值为**该批次平均**后的标量。
- **无梯度/可独立调用**：
  - 指标计算不参与反传，内部应使用 `torch.no_grad()`。
  - 每个指标模块可独立 `import` 使用，例如：
    ```python
    from lib.metrics.psnr import compute_metrics
    metrics = compute_metrics(pred, target)  # {"psnr": 28.34}
    ```
- **批次约定**：默认对 batch 维度做平均；若需要逐样本结果，请在具体实现中提供额外函数（如 `compute_per_image_metrics`），但**统一对外接口仍为** `compute_metrics(pred, target)`。
- **设备与类型**：实现中应支持 CUDA/CPU 自动兼容（与 `pred.device` 对齐），对需要 CPU 的第三方库（如部分 LPIPS/SSIM 实现）需显式迁移并在返回前迁回或转为 `float`。
- **可选的数值规范**（推荐在实现内处理，无需扩展签名）：
  - `data_range`: 依据输入值域自动/显式设定（常见为 `1.0` 或 `2.0`）。
  - `eps`: 数值稳定项，避免 `log(0)`、除零。
  - `convert_colorspace`: 若指标需要特定色彩空间（如 Y 通道），实现内部自行转换。

#### 目录结构建议
```
lib/
  metrics/
    __init__.py
    psnr.py           # compute_metrics -> {"psnr": float}
    ssim.py           # compute_metrics -> {"ssim": float}
    lpips.py          # compute_metrics -> {"lpips": float}
    composite.py      # 聚合调用多个指标并合并字典
```

#### 使用示例
```python
# 单个指标
from lib.metrics.psnr import compute_metrics as psnr_metric
m = psnr_metric(pred, target)        # {"psnr": 28.34}

# 多个指标聚合
from lib.metrics.psnr import compute_metrics as psnr_metric
from lib.metrics.ssim import compute_metrics as ssim_metric

metrics = {}
metrics.update(psnr_metric(pred, target))
metrics.update(ssim_metric(pred, target))
# metrics -> {"psnr": 28.34, "ssim": 0.912}
```

#### 实现要点（约束性建议）
1. **数值稳定**：对 `MSE -> 0` 场景加入 `eps`；对 `log`、`/` 操作做下界限制。  
2. **范围处理**：当输入值域未知时，先尝试自动推断并 `clamp` 到合理区间；必要时在模块顶部提供常量 `DEFAULT_DATA_RANGE`。  
3. **梯度隔离**：统一在函数体最外层包裹 `with torch.no_grad():`。  
4. **跨设备一致**：所有中间张量与 `pred.device` 对齐，返回值尽量转为 Python `float`。  
5. **可测试性**：针对极端输入（全零、全一、随机噪声、pred==target）编写最小单元测试，验证边界行为。  
6. **确定性**：若指标实现可能调用非确定性算子，文档中说明并给出 `torch.use_deterministic_algorithms(True)` 或 `cudnn.deterministic=True` 的指引。  


