"""
DINOv3 ViT-B/16 天气四分类推理入口。

比赛平台约束：
  - CPU 2 核 / 8 GiB 内存
  - 总推理时间 ≤ 70 分钟
  - 测试集约几千张
  - PyTorch 2.17

⚠️ 重要：DINOv3 ViT-B/16 (86M 参数, 384×384 输入) 在 CPU 上较慢，
   请务必提前做 CPU 压测，确认几千张图能在 70 分钟内跑完。
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2

# ==========================================
# 0. CPU 环境优化
# ==========================================
# 匹配比赛平台 2 核配置，避免线程竞争反而变慢
torch.set_num_threads(2)

# ==========================================
# 1. 定位 DINOv3 源码
# ==========================================
_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dinov3-main")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dinov3.hub.backbones import dinov3_vitb16

# ==========================================
# 2. 配置
# ==========================================
CHECKPOINT_PATH = "./finetuned_model_a100_full.pt"
IMG_SIZE = 384

# 比赛平台评测只用 CPU
DEVICE = torch.device("cpu")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ImageFolder 按目录名字母序排列: cloudy, rainy, snowy, sunny
LABELS = ['cloudy', 'rainy', 'snowy', 'sunny']


# ==========================================
# 3. GeM Pooling 模块（与训练完全一致）
# ==========================================
class GeMPooling(nn.Module):
    """Generalized Mean Pooling"""
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)


# ==========================================
# 4. 图像预处理（与训练/验证完全一致）
# ==========================================
preprocess = v2.Compose([
    v2.ToImage(),
    v2.Resize(int(IMG_SIZE * 1.14), interpolation=v2.InterpolationMode.BICUBIC),
    v2.CenterCrop(IMG_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ==========================================
# 5. 加载模型（模块导入时执行一次）
# ==========================================
print(f"[*] Loading checkpoint from: {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

# 5.1 构建 backbone
backbone = dinov3_vitb16(pretrained=False)
embed_dim  = backbone.embed_dim          # 768
hidden_dim = 2 * embed_dim               # 1536

# 5.2 推断类别数
classifier_weight = ckpt["classifier_state_dict"]["1.weight"]
num_classes = classifier_weight.shape[0]
print(f"[*] Detected {num_classes} classes: {LABELS if num_classes == len(LABELS) else 'WARNING: MISMATCH - check LABELS!'}")

# 5.3 分类头
classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(hidden_dim, num_classes)
)
gem_pool = GeMPooling()

# 5.4 加载权重
backbone.load_state_dict(ckpt["backbone_state_dict"])
classifier.load_state_dict(ckpt["classifier_state_dict"])

# 5.5 eval 模式
backbone.eval()
classifier.eval()

# 5.6 torch.compile 说明：
#     理论上可加速 CPU 推理，但首次编译耗时较长（数十秒~数分钟），
#     可能导致平台判为超时。如需启用，取消下面注释即可。
# try:
#     backbone = torch.compile(backbone, mode="reduce-overhead")
#     print("[*] torch.compile enabled")
# except Exception as e:
#     print(f"[*] torch.compile skipped: {e}")

print(f"[*] Model loaded. 参数量: {sum(p.numel() for p in backbone.parameters()) / 1e6:.1f}M")
print(f"[*] WARNING: CPU inference - ensure total time < 70 min for full test set!")


# ==========================================
# 6. 预测函数（平台调用的入口）
# ==========================================
def predict(X):
    """
    模型预测

    param：
        X : np.ndarray，由 cv2.imread 读取的图片数据，shape(224,224,3)。
    return：
        y_predict : str，数据 label，取值为 'sunny', 'cloudy', 'rainy', 'snowy' 之一。
    """
    # 6.1 cv2.imread 读入的是 BGR，转为连续 RGB
    X_rgb = np.ascontiguousarray(X[:, :, ::-1])

    # 6.2 预处理（与训练完全一致）
    img_tensor = preprocess(X_rgb)                  # [3, 384, 384], float32
    img_tensor = img_tensor.unsqueeze(0)             # [1, 3, 384, 384]

    # 6.3 前向推理（CPU float32）
    with torch.no_grad():
        feat_out = backbone.forward_features(img_tensor)
        cls_token = feat_out["x_norm_clstoken"]               # [1, 768]
        patch_gem = gem_pool(feat_out["x_norm_patchtokens"])   # [1, 768]
        feat = torch.cat([cls_token, patch_gem], dim=1)        # [1, 1536]
        logits = classifier(feat)                              # [1, num_classes]

    # 6.4 返回标签
    pred_idx = logits.argmax(dim=1).item()
    return LABELS[pred_idx]


# ==========================================
# 7. CPU 压测工具（可选，开发时使用）
# ==========================================
def benchmark(sample_img_path: str, n_warmup: int = 3, n_test: int = 50):
    """
    快速 CPU 推理速度测试。

    Usage:
        from main_dinov3 import benchmark
        benchmark("test.jpg")
    """
    import cv2
    img = cv2.imread(sample_img_path)
    if img is None:
        print(f"Failed to load: {sample_img_path}")
        return

    # Warmup (includes torch.compile first-time compilation)
    for _ in range(n_warmup):
        predict(img)

    # Timed runs
    times = []
    for _ in range(n_test):
        t0 = time.perf_counter()
        predict(img)
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    print(f"\nCPU 推理压测 ({n_test} 次):")
    print(f"  平均单张耗时: {avg_ms:.1f} ms")
    print(f"  预估 3000 张总耗时: {avg_ms * 3000 / 1000 / 60:.1f} 分钟")
    print(f"  预估 5000 张总耗时: {avg_ms * 5000 / 1000 / 60:.1f} 分钟")
    print(f"  [WARN] Competition limit: <= 70 minutes")
