"""
模型导出脚本 — 将 timm 训练的模型转为 TorchScript 格式
TorchScript 自包含架构，加载时不需要 timm / torchvision
在服务器上运行：python export_model.py
"""
import torch
import timm

MODEL_NAME = "efficientnet_b0"
CHECKPOINT = "./results/efficientnet_b0_final.pth"
OUTPUT = "./results/efficientnet_b0_scripted.pt"
IMG_SIZE = 224
NUM_CLASSES = 4

print(f"[Export] 创建 {MODEL_NAME} 模型...")
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)

print(f"[Export] 加载权重: {CHECKPOINT}")
state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model.load_state_dict(state)
model.eval()

print(f"[Export] 转为 TorchScript...")
example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
traced = torch.jit.trace(model, example)

print(f"[Export] 保存至: {OUTPUT}")
torch.jit.save(traced, OUTPUT)

# 验证可加载
print(f"[Export] 验证...")
loaded = torch.jit.load(OUTPUT)
loaded.eval()
with torch.no_grad():
    out = loaded(example)
    pred = out.argmax(dim=1).item()
    labels = ["cloudy", "rainy", "snowy", "sunny"]
    print(f"[Export] 测试推理: {labels[pred]}")
    print(f"[Export] 输出维度: {out.shape}")

import os
size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
print(f"[Export] 文件大小: {size_mb:.1f} MB")
print(f"[Export] ✅ 完成！将此文件下载到本地 results/ 文件夹")
