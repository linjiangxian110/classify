# 实验 001 — EfficientNet-B0 Baseline

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | `efficientnet_b0` (5.3M) |
| 预训练 | ImageNet (timm) |
| 输入尺寸 | 224×224 |
| Batch Size | 32 |
| Epochs | 30 |
| LR | 1e-3, cosine → 1e-6 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Loss | CrossEntropyLoss + class weights |
| 采样 | shuffle（默认） |
| Label Smoothing | 无 |
| 数据增强 | RandomResizedCrop + HFlip(0.3) + 轻量 ColorJitter |
| 验证划分 | Stratified, 15% |

## 结果

```
Macro F1:   0.9094
Micro F1:   0.9186
Accuracy:   0.9186

类别            Precision     Recall         F1
cloudy           0.9231     0.9174     0.9202
rainy            0.8824     0.8955     0.8889
snowy            0.8871     0.9167     0.9016
sunny            0.9286     0.9254     0.9270
```

最优轮次: epoch 25

## 服务器文件路径

- **权重**: `/mnt/data/lck/code/classify/checkpoints/efficientnet_b0_best.pth`
- **提交权重**: `/mnt/data/lck/code/classify/results/efficientnet_b0_final.pth`
- **混淆矩阵**: `/mnt/data/lck/code/classify/results/efficientnet_b0_confusion_matrix.png`

## 分析

- sunny/cloudy 表现好（数据量充足）
- rainy 最弱（446 张），snowy 次弱（403 张）
- 小类是瓶颈，需针对性处理
