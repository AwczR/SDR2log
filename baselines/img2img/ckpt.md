```markdown
# /ckpt 目录结构说明（最简版）

```text
/ckpt/
  └─ {date}-{exp}/
      ├─ config/
      │   ├─ data.yaml          # 数据与色彩空间配置（SDR→ACEScct）
      │   ├─ model.yaml         # 模型结构与损失权重
      │   └─ optim.yaml         # 优化器、学习率计划、batch size 等
      ├─ meta/
      │   └─ env.json           # 依赖版本、GPU、随机种子、git commit
      ├─ checkpoints/
      │   ├─ last.pt
      │   └─ best@psnr_acescct.pt
      ├─ logs/
      │   ├─ scalars.train.jsonl   # {"epoch":1,"loss_total":0.024,"lr":1e-4}
      │   └─ scalars.val.jsonl     # {"epoch":1,"psnr_acescct":38.7,"ssim_acescct":0.967,"mae_acescct":0.012}
      ├─ eval/
      │   └─ summary.json          # 各指标均值±std
      └─ samples/
          ├─ epoch_0005/
          │   ├─ img_001_input.png
          │   ├─ img_001_pred.png
          │   ├─ img_001_gt.png
          │   └─ img_001_diff.png
          ├─ epoch_0010/
          └─ ...
```

## 记录指标
- **训练阶段**：`loss_total`, 子损失（如 `loss_l1`）, `lr`, `iter_s`
- **验证阶段**：`psnr_acescct`, `ssim_acescct`, `mae_acescct`
- **主评测指标**：PSNR\_ACEScct（主），SSIM\_ACEScct，MAE\_ACEScct（辅）

## 采样策略
- 每 N 个 epoch（如 5）在固定验证样本上输出：
  - 输入 SDR（`*_input.png`）
  - 输出 ACEScct（`*_pred.png`）
  - GT ACEScct（`*_gt.png`）
  - 误差图（`*_diff.png`）
```
