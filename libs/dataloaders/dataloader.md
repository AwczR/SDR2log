# dataloader 接口规范

## 模块职责
`dataloader.py` 是整个数据层的**统一入口**。  
它负责根据配置构建合适的 `DataLoader`（图像或视频模式），并返回满足统一协议的迭代器。  
模块内部可调度不同的 Dataset 实现、读取逻辑、增强与色彩转换，但这些对外透明。

---

## 对外暴露的唯一函数

### `build_dataloader(cfg: dict, split: str, mode: str) -> torch.utils.data.DataLoader`
**描述：**  
构建一个可迭代的 PyTorch `DataLoader`，根据配置返回图像或视频样本。  

#### 参数
| 名称 | 类型 | 说明 |
|------|------|------|
| `cfg` | `dict` | 配置字典。应包含至少 `data` 字段，描述路径、批大小、并行数、采样策略等。 |
| `split` | `str` | `"train"`, `"val"`, `"test"` 之一。用于选择对应的数据划分。 |
| `mode` | `str` | `"img"` 或 `"vid"`，指定加载图像对或视频对。<br>若当前数据集不支持 `vid` 模式，必须抛出 `NotImplementedError`。 |

#### 返回
一个 `torch.utils.data.DataLoader` 对象，迭代返回以下两种样本格式之一：

---

## 样本格式（Dataset 输出协议）

### 图像模式 (`mode="img"`)
每个样本应为字典：
```python
{
  "sdr": FloatTensor[3,H,W],   # SDR 输入帧，float32，[0,1]，目标色彩空间（如 ACEScct）
  "hdr": FloatTensor[3,H,W],   # HDR 目标帧，float32，[0,1]，同空间
  "meta": {
    "dataset": str,            # 数据集标识
    "is_video": False,
    "path_sdr": str,           # 文件路径
    "path_hdr": str,           # 文件路径
    "size": (H, W),
    "space_in": str,           # 源空间（例如 'ACEScct'）
    "space_out": str,          # 目标空间（例如 'ACEScct'）
  }
}
```

### 视频模式 (`mode="vid"`)
每个样本应为字典：
```python
{
  "sdr_clip": FloatTensor[T,3,H,W],   # SDR 帧序列 [0,1]
  "hdr_clip": FloatTensor[T,3,H,W],   # HDR 帧序列 [0,1]
  "meta": {
    "dataset": str,
    "is_video": True,
    "video_id": str|int,
    "frame_start": int,
    "clip_len": int,                  # = T
    "size": (H, W),
    "path_sdr": str,
    "path_hdr": str,
    "space_in": str,
    "space_out": str,
  }
}
```

> 若模型仅支持帧级输入（例如 Diff-HDRTV），训练时可将视频样本展平为帧级批次。  
> 但 DataLoader 层仍需提供时序维度，以支持后续扩展。

---

## 异常约定

| 情况 | 异常类型 | 说明 |
|------|-----------|------|
| 请求视频模式，但数据集不支持 | `NotImplementedError` | 必须抛出，避免返回空 DataLoader。 |
| 索引文件或根目录缺失 | `FileNotFoundError` | 初始化阶段检查。 |
| 样本读取失败或尺寸非法 | `ValueError` | 可在 Dataset 内部抛出。 |

---

## 数据规范约束

1. **数值范围**：所有张量必须在 `[0,1]`。  
2. **空间对齐**：`H`, `W` 必须为偶数。  
3. **色彩空间**：输出张量必须已转换到目标训练空间（如 `ACEScct`），模型无需再做色彩变换。  
4. **返回类型**：所有图像、视频帧均为 `torch.FloatTensor`。  
5. **批内一致性**：同一批次样本的尺寸 `(H,W)` 必须一致。  

---

## 示例（使用方式）

```python
from lib.data.dataloader import build_dataloader

cfg = {
    "data": {
        "root": "/data/ACES_pairs",
        "batch_size": 4,
        "num_workers": 8,
        "augment": {"train_crop_size": [320,320], "hflip": True}
    }
}

# 构建图像对加载器
train_loader = build_dataloader(cfg, split="train", mode="img")

# 构建视频对加载器（若数据集支持）
val_loader = build_dataloader(cfg, split="val", mode="vid")
```

---

## 模块内推荐的最小结构（仅说明，不要求实现）
```
lib/data/
  dataloader.py          # 对外暴露 build_dataloader()
  datasets/
    base.py              # 抽象 Dataset 基类，定义通用属性
    aces_pair_img.py     # 图像对实现
    aces_pair_vid.py     # 视频对实现（不支持时 raise）
  pipelines/
    color.py             # 色彩 / EOTF 统一
    augment.py           # 数据增强
```

---

**总结：**  
`dataloader.py` 向外只提供一个函数 `build_dataloader(cfg, split, mode)`。  
它必须在任何实现下返回结构一致的样本字典，并在视频模式不被支持时显式抛出 `NotImplementedError`。
