"""
推理入口 — 适配官方平台提交格式
函数签名 predict(X) 必须保持不变，平台会循环调用此函数
"""
import torch
import numpy as np
import cv2

from config import (
    MODEL_NAME, NUM_CLASSES, IMG_SIZE, RESIZE_VAL,
    IMAGENET_MEAN, IMAGENET_STD, LABELS, DEVICE,
)
from model_zoo import create_model

# ============================================================
# 全局：加载一次模型（模块导入时执行，避免每次 predict 重新加载）
# ============================================================
WEIGHT_PATH = f"./results/{MODEL_NAME}_final.pth"

_model = None


def _get_model():
    """懒加载模型（首次调用时加载权重）"""
    global _model
    if _model is None:
        print(f"[Infer] 加载模型: {MODEL_NAME}")
        print(f"[Infer] 权重路径: {WEIGHT_PATH}")
        _model = create_model(MODEL_NAME, NUM_CLASSES, pretrained=False)
        _model.load_state_dict(
            torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
        )
        _model.to(DEVICE)
        _model.eval()
        print(f"[Infer] 模型加载完成，设备: {DEVICE}")
    return _model


def _preprocess(X):
    """
    图片预处理 — 必须与训练时的验证预处理完全一致：
    Resize(256) → CenterCrop(224) → Normalize(ImageNet)
    """
    # 统一到 256x256
    X = cv2.resize(X, (RESIZE_VAL, RESIZE_VAL))

    # CenterCrop 到 224x224
    h, w = X.shape[:2]
    top = (h - IMG_SIZE) // 2
    left = (w - IMG_SIZE) // 2
    X = X[top:top + IMG_SIZE, left:left + IMG_SIZE]

    # HWC → CHW, float32, 归一化到 [0, 1]
    X = X.astype(np.float32) / 255.0
    X = np.transpose(X, (2, 0, 1))

    # ImageNet 标准化
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    X = (X - mean) / std

    # 加 batch 维度
    X = X[np.newaxis, :, :, :]
    return torch.from_numpy(X)


def predict(X):
    """
    模型预测（官方平台调用接口）

    Args:
        X: np.ndarray，cv2.imread 读取的图片，shape (H, W, 3)

    Returns:
        y_predict: str，取值为 'cloudy', 'rainy', 'snowy', 'sunny' 之一
    """
    model = _get_model()

    # 预处理
    tensor = _preprocess(X).to(DEVICE)

    # 推理
    with torch.no_grad():
        output = model(tensor)
        pred_idx = int(torch.argmax(output, dim=1).item())

    return LABELS[pred_idx]


# ============================================================
# 本地测试
# ============================================================
if __name__ == "__main__":
    import os
    import time

    # 从训练数据中随机找几张图片测试
    test_dir = "./train"
    test_images = []
    for root, dirs, files in os.walk(test_dir):
        for f in files[:3]:  # 每类取 3 张
            if f.endswith((".jpg", ".png", ".jpeg")):
                test_images.append(os.path.join(root, f))
        if len(test_images) >= 10:
            break

    if not test_images:
        print("[Test] 未找到测试图片，请确认 train/ 目录存在")
    else:
        print(f"[Test] 测试 {len(test_images)} 张图片...\n")
        total_time = 0.0
        for path in test_images:
            img = cv2.imread(path)
            if img is None:
                print(f"  ⚠ 无法读取: {path}")
                continue

            t0 = time.time()
            result = predict(img)
            elapsed = time.time() - t0
            total_time += elapsed

            print(f"  {os.path.basename(path):30s} → {result:<8s}  ({elapsed:.3f}s)")

        avg = total_time / len(test_images)
        print(f"\n  平均单张耗时: {avg:.3f}s")
        print(f"  估计 3000 张耗时: {avg * 3000 / 60:.1f} 分钟")
