# `libs/color` 模块文档（接口规范）

## 1. 模块职责
`libs/color` 作为**颜色与数值域的唯一入口**，向数据层/模型层提供**读取、归一化、尺寸规范化、LUT 应用**与**色彩/EOTF 转换**等能力。  
上层不得直接调用第三方图像库或自行做色彩数学，需通过本模块解耦。

---

## 2. 目录结构（建议）
```
libs/color/
  __init__.py        # 汇总对外API
  io.py              # 影像读写与数值归一化
  lut.py             # .cube 3D LUT 解析与应用
  ops.py             # 尺寸规范化、Tensor 封装、空间转换路由
```

---

## 3. 统一约定
- **数据范围**：模块所有输入/输出影像均为 `float32` 且在 **[0,1]**。
- **排列**：函数默认接收/返回 `H×W×3`（RGB），提供 `to_chw_tensor()` 将其转为 `3×H×W`。
- **尺寸**：`H,W` 最终应为**偶数**；若不满足，需通过 `ensure_even_hw()` 纠正。
- **空间标签**：使用明确字符串标识，示例：`"LogC3" | "HLG" | "PQ" | "Rec709" | "ACEScct" | "ACEScg" | "AWG3"`。
- **异常**：文件不存在 → `FileNotFoundError`；不支持/格式错误 → `NotImplementedError`；数值/形状非法 → `ValueError`。

---

## 4. 对外 API

### 4.1 读取与归一化
```python
read_tiff_as_float01(path: str) -> np.ndarray[H, W, 3], float32 in [0,1]
```
**功能**：读取 `.tif/.tiff`，如为整型（10/12/16-bit 等）按位深线性缩放到 `[0,1]`；若为浮点，裁剪到 `[0,1]`。  
**行为**：
- 灰度 → 三通道复制；带 alpha → 丢弃 alpha。  
**异常**：`FileNotFoundError`、`ValueError`（通道/形状不支持）。

---

### 4.2 LUT 解析与应用
```python
load_cube_lut(path: str) -> dict
# 返回：
# {
#   "table": np.ndarray[L, L, L, 3], float32 in [0,1],
#   "domain_min": np.ndarray[3], float32,
#   "domain_max": np.ndarray[3], float32,
# }
```
**功能**：解析 3D `.cube`（要求存在 `LUT_3D_SIZE`，可选 `DOMAIN_MIN/MAX`）。  
**异常**：无 `LUT_3D_SIZE`/维度不匹配 → `NotImplementedError`/`ValueError`。

```python
apply_3d_lut(img: np.ndarray[H, W, 3], lut: dict, mode: str="trilinear") -> np.ndarray[H, W, 3]
```
**功能**：将 `img`（[0,1]）映射进 LUT 域（考虑 `DOMAIN_MIN/MAX`）后采样：  
- `mode="trilinear"`：三线性插值（默认）；`"nearest"`：最近邻索引。  
**异常**：形状/范围非法 → `ValueError`。

---

### 4.3 尺寸规范化与张量封装
```python
ensure_even_hw(img: np.ndarray, how: str="center_crop") -> np.ndarray
```
**功能**：确保 `H,W` 为偶数；  
- `how="center_crop"`：自中心裁去 1px 边；  
- `how="floor"`：直接下取偶数尺寸。  

```python
to_chw_tensor(img: np.ndarray) -> torch.FloatTensor[3, H, W]
```
**功能**：`H×W×3 (float32,[0,1])` → PyTorch `FloatTensor`，通道前置，内存连续。

---

### 4.4 色彩/EOTF 转换路由
```python
convert_space(img: np.ndarray, src: str, dst: str, meta: dict|None=None) -> np.ndarray
```
**功能**：统一的色彩/EOTF 入口，用于在不同记忆色空间/工作空间之间转换（如 `LogC3/AWG3 → ACEScct`、`HLG/PQ → ACEScct`、`Rec709 ↔ ACEScct` 等）。  
**当前最小实现**：`src==dst` 返回原图；其他路径若未实现抛 `NotImplementedError`。  
**扩展点**（实现要求）：
- **EOTF/OETF**：HLG、PQ 的正/逆变换；
- **基色矩阵**：AWG3 ↔ ACEScg/Rec.709（基于原/目标色度与白点）；
- **对数编码**：LogC3 正/逆函数；
- **位深安全域**：全流程保持 `float32,[0,1]`，必要处裁剪；
- **元数据**：`meta` 可传相机型号、白点、曝光补偿、LUT 名称等，便于可重复性记录。

---

## 5. 典型调用序（示例性的流程约定）
1. `img = read_tiff_as_float01(path)`  
2. （可选）`img = convert_space(img, src="LogC3", dst="ACEScct")`  
3. （可选）`img_sdr = apply_3d_lut(img_or_linear, lut)`  
4. `img = ensure_even_hw(img)`（与 `img_sdr` 同步处理）  
5. `t = to_chw_tensor(img)`  

> 注意：若上层需要批内尺寸一致，应在裁剪参数上保持**同步**（同一随机裁剪应用于成对影像）。

---

## 6. 错误与边界条件
- **文件缺失/不可读** → `FileNotFoundError`；  
- **不支持的 .cube（非 3D、缺表、条目数不匹配）** → `NotImplementedError`/`ValueError`；  
- **影像形状不合规（通道数≠3/4，维度≠2/3）** → `ValueError`；  
- **色彩路径未实现** → `NotImplementedError`（由上层决定降级或中止）。

---

## 7. 性能与数值稳定性建议
- LUT 应用：优先向量化，避免逐像素 Python 循环；必要时可引入 chunking 以控制峰值内存。  
- 避免重复装载 LUT：在上层 Dataset 初始化阶段缓存 `load_cube_lut()` 结果。  
- 坚持 `float32` 全链路，最后一步统一裁剪到 `[0,1]`。  
- 对 EOTF/OETF 变换使用**连续可导**实现（训练中需要反传时用 PyTorch 对应实现）。

---

## 8. 可测试性（最小断言）
- I/O：整型 tiff 的 `max()≈1.0`、`min()≥0.0`；浮点 tiff 裁剪正确。  
- LUT：随机输入恒定时，`apply_3d_lut` 输出稳定且在 `[0,1]`；`DOMAIN_MIN/MAX` 生效。  
- 尺寸：`ensure_even_hw` 对任意奇偶组合都返回偶数尺寸。  
- 路由：`convert_space(src==dst)` 恒等；未实现路径抛 `NotImplementedError`。

---

## 9. 版本与兼容
- **语义版本**：`MAJOR.MINOR.PATCH`；  
- **破坏性更新**仅发生在 `MAJOR` 变更（如 API 签名变化）；  
- 对新增色彩路径在 `MINOR` 升级中加入，不改变现有行为。
