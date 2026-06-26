# 实验 002 — EfficientNet-B0 + 均衡采样 + Label Smoothing

## 相比 exp_001 的改动

| 改动 | 旧值 | 新值 |
|------|------|------|
| 采样方式 | shuffle | **WeightedRandomSampler**（每 batch 四类均衡） |
| Label Smoothing | 无 | **0.1** |
| Epochs | 30 | **50** |

相关代码文件：
- `config.py` — 新增 `USE_BALANCED_SAMPLER=True`, `LABEL_SMOOTHING=0.1`, `EPOCHS=50`
- `dataset.py` — 新增 `WeightedRandomSampler` 分支
- `train.py` — `CrossEntropyLoss` 加入 `label_smoothing` 参数

## 结果

```
Macro F1:   0.9059  (↓ -0.0035)
Micro F1:   0.9239
Accuracy:   0.9239

类别            Precision     Recall         F1
cloudy           0.9578     0.9021     0.9291  (↑)
rainy            0.8286     0.8657     0.8467  (↓ -0.0422!)
snowy            0.9016     0.9167     0.9091  (↑)
sunny            0.9161     0.9627     0.9388  (↑)
```

最优轮次: epoch 11（过早收敛）

## 服务器文件路径

- **权重**: `/mnt/data/lck/code/classify/checkpoints/efficientnet_b0_best.pth`
- **提交权重**: `/mnt/data/lck/code/classify/results/efficientnet_b0_final.pth`

## 诊断

**均衡采样起了反效果**：
- rainy Precision 从 0.8824 → 0.8286（-0.054），模型把更多 cloudy/snowy 误判为 rainy
- 均衡采样导致 rainy 的 446 张图被反复采样 → **过拟合训练集中的 rainy 样本** → 泛化变差
- 收敛过快（epoch 11 vs 25）也印证了过拟合

**结论**：WeightedRandomSampler 对小数据集的小类不友好，下个实验移除。
