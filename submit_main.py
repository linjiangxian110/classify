"""
MO 平台提交文件 — main.py
使用 TorchScript 模型，无需 timm / torchvision
"""
import torch
import numpy as np
import cv2

# ============================================================
# 配置（与训练时一致，勿修改）
# ============================================================
LABELS = ["cloudy", "rainy", "snowy", "sunny"]
IMG_SIZE = 224
RESIZE_VAL = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================
# 加载模型（修改权重路径即可）
# ============================================================
MODEL_PATH = "./results/efficientnet_b0_scripted.pt"
DEVICE = torch.device("cpu")

_model = torch.jit.load(MODEL_PATH, map_location="cpu")
_model.eval()
print(f"[Model] TorchScript 模型已加载: {MODEL_PATH}")


def predict(X):
    """
    模型预测（官方平台调用接口 — 勿改函数签名）

    Args:
        X: np.ndarray，cv2.imread 读取的图片，shape (H, W, 3)

    Returns:
        str: 类别标签，'cloudy', 'rainy', 'snowy', 'sunny' 之一
    """
    # ── 预处理（与训练验证完全一致）──
    # Resize → CenterCrop → Normalize(ImageNet)
    X = cv2.resize(X, (RESIZE_VAL, RESIZE_VAL))

    h, w = X.shape[:2]
    top = (h - IMG_SIZE) // 2
    left = (w - IMG_SIZE) // 2
    X = X[top:top + IMG_SIZE, left:left + IMG_SIZE]

    X = X.astype(np.float32) / 255.0
    X = np.transpose(X, (2, 0, 1))

    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    X = (X - mean) / std

    X = X[np.newaxis, :, :, :]
    X = torch.from_numpy(X).to(DEVICE)

    # ── 推理 ──
    with torch.no_grad():
        output = _model(X)
        pred_idx = int(torch.argmax(output, dim=1).item())

    return LABELS[pred_idx]
