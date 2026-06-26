"""
模型工厂 — 通过 timm 统一加载预训练模型
支持: EfficientNet / ResNet / MobileNetV3 / ConvNeXt
"""
import torch.nn as nn
import timm

# 支持的模型列表及其输入尺寸建议
SUPPORTED_MODELS = {
    "efficientnet_b0": {
        "input_size": 224,
        "family": "EfficientNet",
        "params_m": 5.3,
    },
    "efficientnet_b1": {
        "input_size": 240,
        "family": "EfficientNet",
        "params_m": 7.8,
    },
    "resnet18": {
        "input_size": 224,
        "family": "ResNet",
        "params_m": 11.7,
    },
    "resnet34": {
        "input_size": 224,
        "family": "ResNet",
        "params_m": 21.8,
    },
    "mobilenetv3_large_100": {
        "input_size": 224,
        "family": "MobileNetV3",
        "params_m": 5.4,
    },
    "convnext_tiny": {
        "input_size": 224,
        "family": "ConvNeXt",
        "params_m": 28.6,
    },
}


def create_model(model_name="efficientnet_b0", num_classes=4, pretrained=True):
    """
    创建模型（ImageNet 预训练 + 替换分类头）

    Args:
        model_name: timm 模型名，如 "efficientnet_b0"
        num_classes: 分类数，默认 4
        pretrained: 是否加载 ImageNet 预训练权重

    Returns:
        model: nn.Module
    """
    if model_name not in SUPPORTED_MODELS:
        available = ", ".join(SUPPORTED_MODELS.keys())
        raise ValueError(
            f"不支持的模型: {model_name}\n"
            f"可选模型: {available}"
        )

    info = SUPPORTED_MODELS[model_name]
    print(f"[Model] 加载 {model_name} "
          f"({info['family']}, {info['params_m']}M 参数)")
    print(f"[Model] pretrained={pretrained}, num_classes={num_classes}")

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    # 统计参数量
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 总参数: {total:,}  |  可训练: {trainable:,}")
    print(f"[Model] 建议输入尺寸: {info['input_size']}×{info['input_size']}")

    return model


def get_recommended_input_size(model_name):
    """返回模型推荐的输入尺寸"""
    return SUPPORTED_MODELS.get(model_name, {}).get("input_size", 224)


if __name__ == "__main__":
    # 快速测试
    for name in ["efficientnet_b0", "resnet18",
                 "mobilenetv3_large_100", "convnext_tiny"]:
        print(f"\n{'='*50}")
        m = create_model(name, num_classes=4, pretrained=False)
        print(f"输出维度验证: {m(torch.randn(2, 3, 224, 224)).shape}")
        del m
