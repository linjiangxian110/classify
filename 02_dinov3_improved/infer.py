"""
DINOv3 TTA 推理 — MO 平台提交用 main.py 内容
改进：水平翻转 TTA → 两张图取平均概率
对比无 TTA 版：预期 +0.5~1 点 Macro F1
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2

# ── 路径 ──
_DINOV3 = os.path.join(os.path.dirname(__file__), "dinov3-main")
if _DINOV3 not in sys.path:
    sys.path.insert(0, _DINOV3)
from dinov3.hub.backbones import dinov3_vitb16

torch.set_num_threads(2)

# ── 配置 ──
CHECKPOINT_PATH = "./finetuned_model_ema.pt"   # EMA 权重
IMG_SIZE = 384
RESIZE_VAL = int(IMG_SIZE * 1.14)               # 438
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
LABELS = ["cloudy", "rainy", "snowy", "sunny"]

# ── GeM Pooling ──
class GeMPool(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)

# ── 预处理（无翻转）──
_preprocess = v2.Compose([
    v2.ToImage(),
    v2.Resize(RESIZE_VAL, interpolation=v2.InterpolationMode.BICUBIC),
    v2.CenterCrop(IMG_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── 预处理（水平翻转）──
class FlipPreprocess:
    """水平翻转预处理：处理 → flip dim=-1 → 转回"""
    def __call__(self, img):
        t = _preprocess(img)
        return torch.flip(t, dims=[-1])


_flip_preprocess = FlipPreprocess()

# ── 加载模型 ──
print(f"[*] Loading: {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
num_classes = ckpt["classifier_state_dict"]["1.weight"].shape[0]

backbone = dinov3_vitb16(pretrained=False)
backbone.load_state_dict(ckpt["backbone_state_dict"])
backbone.eval()

gem_pool = GeMPool()
classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(2 * backbone.embed_dim, num_classes),
)
classifier.load_state_dict(ckpt["classifier_state_dict"])
classifier.eval()

print(f"[*] Model ready. Classes: {num_classes}")


def _predict_single(x_tensor):
    """单次前向推理，返回 logits tensor [1, num_classes]"""
    with torch.no_grad():
        feat = backbone.forward_features(x_tensor)
        cls_token = feat["x_norm_clstoken"]
        patch_gem = gem_pool(feat["x_norm_patchtokens"])
        fused = torch.cat([cls_token, patch_gem], dim=1)
        return classifier(fused)


def predict(X):
    """
    TTA 预测 — 原图 + 水平翻转取平均
    """
    # BGR → RGB
    X_rgb = np.ascontiguousarray(X[:, :, ::-1])

    # 原图
    t_orig = _preprocess(X_rgb).unsqueeze(0)                # [1, 3, 384, 384]
    # 翻转
    t_flip = torch.flip(t_orig, dims=[-1])                  # [1, 3, 384, 384]

    logit_orig = _predict_single(t_orig)
    logit_flip = _predict_single(t_flip)

    # 平均
    logit_avg = (logit_orig + logit_flip) / 2.0
    pred_idx = logit_avg.argmax(dim=1).item()
    return LABELS[pred_idx]


# ============================================================
# CPU 压测
# ============================================================
if __name__ == "__main__":
    import time, cv2

    # 测试图片路径
    test_img = os.path.join(
        os.path.dirname(__file__), "..", "train", "cloudy", "cloudy_00001.jpg"
    )
    if not os.path.exists(test_img):
        test_img = os.path.join(
            os.path.dirname(__file__), "..", "train", "sunny", "sunny_00009.jpg"
        )

    if os.path.exists(test_img):
        img = cv2.imread(test_img)
        print(f"测试图片: {test_img}")

        # Warmup
        for _ in range(3):
            predict(img)

        # 计时
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            predict(img)
            times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        print(f"\nTTA 推理压测 (20次):")
        print(f"  平均单张: {avg_ms:.1f} ms")
        print(f"  3000 张:   {avg_ms * 3000 / 60000:.1f} 分钟")
        print(f"  5000 张:   {avg_ms * 5000 / 60000:.1f} 分钟")
    else:
        print("测试图片不存在，跳过压测")
