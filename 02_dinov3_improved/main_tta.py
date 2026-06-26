"""
MO 平台提交 — DINOv3 + TTA（水平翻转取平均）
在原版 94 分模型基础上只加 TTA，预期 +0.5~1 点
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2

torch.set_num_threads(2)

# ── dinov3 路径 ──
_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dinov3-main")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from dinov3.hub.backbones import dinov3_vitb16

# ── 配置 ──
CHECKPOINT_PATH = "./finetuned_model_a100_full.pt"
IMG_SIZE = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
LABELS = ['cloudy', 'rainy', 'snowy', 'sunny']


class GeMPool(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)


# ── 预处理 ──
preprocess = v2.Compose([
    v2.ToImage(),
    v2.Resize(int(IMG_SIZE * 1.14), interpolation=v2.InterpolationMode.BICUBIC),
    v2.CenterCrop(IMG_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── 加载模型 ──
print(f"[*] 加载: {CHECKPOINT_PATH}")
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

print(f"[*] 模型就绪, {num_classes} 类 [TTA 模式]")


@torch.no_grad()
def _forward(tensor):
    feat = backbone.forward_features(tensor)
    cls_token = feat["x_norm_clstoken"]
    patch_gem = gem_pool(feat["x_norm_patchtokens"])
    fused = torch.cat([cls_token, patch_gem], dim=1)
    return classifier(fused)


def predict(X):
    """
    TTA 推理：原图 + 水平翻转 → 两次推理取平均
    """
    # BGR → RGB
    X_rgb = np.ascontiguousarray(X[:, :, ::-1])

    # 原图
    t_orig = preprocess(X_rgb).unsqueeze(0)
    # 水平翻转
    t_flip = torch.flip(t_orig, dims=[-1])

    # 两次推理取平均
    logit_orig = _forward(t_orig)
    logit_flip = _forward(t_flip)
    logit_avg = (logit_orig + logit_flip) / 2.0

    return LABELS[logit_avg.argmax(dim=1).item()]
