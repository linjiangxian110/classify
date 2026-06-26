"""
CPU 推理压测脚本
模拟平台评测环境（CPU 推理），估算总耗时
"""
import os
import time
import argparse
import numpy as np
import cv2
import torch

from config import MODEL_NAME, NUM_CLASSES, LABELS, RESULT_DIR
from model_zoo import create_model


def load_model(weight_path):
    """加载模型到 CPU"""
    print(f"[Bench] 加载模型: {MODEL_NAME}")
    print(f"[Bench] 权重: {weight_path}")
    model = create_model(MODEL_NAME, NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=False))
    model.to("cpu")
    model.eval()

    # 统计
    total = sum(p.numel() for p in model.parameters())
    param_size_mb = total * 4 / (1024 * 1024)  # float32 → MB
    print(f"[Bench] 参数量: {total:,}")
    print(f"[Bench] 权重体积(估算): {param_size_mb:.1f} MB")
    return model


def benchmark(model, image_size=224, num_runs=100, num_warmup=10):
    """CPU 推理压测"""
    print(f"\n[Bench] 压测配置: {num_runs} 次推理, 输入 {image_size}×{image_size}")

    # 构造随机输入（模拟真实图片）
    dummy = torch.randn(1, 3, image_size, image_size)

    # 预热（首次推理包含模型初始化开销）
    print(f"[Bench] 预热 {num_warmup} 次...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(dummy)

    # 正式计时
    print(f"[Bench] 正式计时 {num_runs} 次...")
    torch.cpu.synchronize() if hasattr(torch.cpu, "synchronize") else None
    t0 = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy)
    elapsed = time.time() - t0

    avg_ms = (elapsed / num_runs) * 1000
    avg_s = elapsed / num_runs

    print(f"\n{'='*50}")
    print(f"  总耗时:      {elapsed:.2f}s ({num_runs} 次)")
    print(f"  单张平均:    {avg_ms:.1f} ms ({avg_s:.4f}s)")
    print(f"{'='*50}")
    print(f"  估算 2000 张: {avg_s * 2000 / 60:.1f} 分钟")
    print(f"  估算 3000 张: {avg_s * 3000 / 60:.1f} 分钟")
    print(f"  估算 5000 张: {avg_s * 5000 / 60:.1f} 分钟")
    print(f"  70 分钟上限:  ~{int(70 * 60 / avg_s)} 张")
    print(f"{'='*50}")

    # 判断
    if avg_s * 5000 / 60 <= 70:
        print("  ✅ 安全：5000 张可在 70 分钟内完成")
    elif avg_s * 3000 / 60 <= 70:
        print("  ⚠️  可接受：3000 张可在 70 分钟内完成")
    else:
        print("  ❌ 危险：可能超时，建议换更轻的模型")


def main():
    parser = argparse.ArgumentParser(description="CPU 推理压测")
    parser.add_argument(
        "--weight", type=str,
        default=f"./results/{MODEL_NAME}_final.pth",
        help="权重文件路径",
    )
    parser.add_argument("--runs", type=int, default=100,
                        help="推理次数（默认 100）")
    parser.add_argument("--size", type=int, default=224,
                        help="输入尺寸（默认 224）")
    args = parser.parse_args()

    model = load_model(args.weight)
    benchmark(model, image_size=args.size, num_runs=args.runs)


if __name__ == "__main__":
    main()
