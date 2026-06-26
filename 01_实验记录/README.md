# 01 — 实验记录索引

> 记录每次训练的配置、结果和权重路径，便于复现和对比。

---

## 实验列表

| 编号 | 日期 | 模型 | 关键改动 | Macro F1 | 权重路径 |
|------|------|------|----------|----------|----------|
| [001](exp_001_efficientnet_b0_baseline.md) | 06-26 | EfficientNet-B0 | baseline（class weights + cosine lr） | **0.9094** | `checkpoints/efficientnet_b0_best.pth` |
| [002](exp_002_efficientnet_b0_balanced_sampler.md) | 06-26 | EfficientNet-B0 | +均衡采样 + label smoothing | **0.9059** ↓ | `checkpoints/efficientnet_b0_best.pth` |
| 003 | — | ConvNeXt-Tiny | 计划中 | — | — |

---

## 最佳模型（当前）

- **Macro F1**: 0.9094（exp_001）
- **权重**: `/mnt/data/lck/code/classify/checkpoints/efficientnet_b0_best.pth`
- **提交权重**: `/mnt/data/lck/code/classify/results/efficientnet_b0_final.pth`
