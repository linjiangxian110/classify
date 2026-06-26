"""
数据加载与预处理 — 训练/验证/推理统一处理管线
"""
import os
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from config import (
    DATA_DIR, IMG_SIZE, RESIZE_VAL, BATCH_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, VAL_RATIO, SEED,
    NUM_CLASSES, USE_BALANCED_SAMPLER,
)


def get_train_transforms():
    """训练增强：随机裁剪 + 水平翻转 + 轻量颜色抖动"""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    """验证/推理预处理：缩放 + 中心裁剪（不做增强）"""
    return transforms.Compose([
        transforms.Resize((RESIZE_VAL, RESIZE_VAL)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_loaders(data_dir=None, batch_size=None, val_ratio=None,
                  stratified=True):
    """
    构建训练和验证 DataLoader。

    Args:
        data_dir: 数据目录，默认从 config.DATA_DIR 读取
        batch_size: batch 大小
        val_ratio: 验证集比例
        stratified: 是否分层划分（保证各类比例一致），推荐 True

    Returns:
        train_loader, val_loader, class_names
    """
    data_dir = data_dir or DATA_DIR
    batch_size = batch_size or BATCH_SIZE
    val_ratio = val_ratio or VAL_RATIO

    # 完整数据集（带训练增强的版本用于训练，验证版本用于评估）
    full_train_set = datasets.ImageFolder(data_dir, transform=get_train_transforms())
    full_val_set = datasets.ImageFolder(data_dir, transform=get_val_transforms())
    class_names = full_train_set.classes

    print(f"[Data] 数据目录: {data_dir}")
    print(f"[Data] 类别: {class_names}")
    print(f"[Data] 各类数量: {[f'{n}: {class_names[i]}' for i, n in enumerate(np.bincount(full_train_set.targets))]}")

    n_val = int(len(full_train_set) * val_ratio)
    n_train = len(full_train_set) - n_val

    if stratified:
        # 分层划分，保证验证集中各类比例与总体一致
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=n_val, random_state=SEED
        )
        train_indices, val_indices = next(
            sss.split(np.zeros(len(full_train_set)), full_train_set.targets)
        )
        train_set = torch.utils.data.Subset(full_train_set, train_indices)
        val_set = torch.utils.data.Subset(full_val_set, val_indices)

        val_targets = [full_train_set.targets[i] for i in val_indices]
        val_dist = np.bincount(val_targets, minlength=NUM_CLASSES)
        print(f"[Data] 验证集各类数量: {list(val_dist)}")
    else:
        train_set, val_set = random_split(
            full_train_set, [n_train, n_val],
            generator=torch.Generator().manual_seed(SEED),
        )

    print(f"[Data] 训练集: {n_train} 张  |  验证集: {n_val} 张")

    # ── 构建训练 DataLoader ──
    if USE_BALANCED_SAMPLER:
        # 获取训练集中每个样本的标签
        train_targets = [full_train_set.targets[i] for i in train_indices]
        class_counts = np.bincount(train_targets, minlength=NUM_CLASSES)
        # 每类权重 = 1/该类样本数，样本被采样概率反比于类别频率
        class_weights_sampler = 1.0 / class_counts
        sample_weights = class_weights_sampler[train_targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_targets),
            replacement=True,
        )
        train_loader = DataLoader(
            train_set, batch_size=batch_size,
            sampler=sampler,
            num_workers=0, pin_memory=True,
        )
        print(f"[Data] 使用 WeightedRandomSampler（均衡采样）")
    else:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True,
        )

    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    return train_loader, val_loader, class_names


def compute_class_weights(data_dir=None):
    """
    计算类别权重（用于 CrossEntropyLoss）。
    公式: weight = total_samples / (num_classes * samples_per_class)
    小类权重大，大类权重小。
    """
    data_dir = data_dir or DATA_DIR
    full_set = datasets.ImageFolder(data_dir)
    targets = np.array(full_set.targets)
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)
    total = class_counts.sum()
    weights = total / (NUM_CLASSES * class_counts)
    weights = torch.tensor(weights, dtype=torch.float32)

    print(f"[Data] 类别计数: {dict(zip(full_set.classes, class_counts.tolist()))}")
    print(f"[Data] 类别权重: {dict(zip(full_set.classes, weights.tolist()))}")

    return weights


if __name__ == "__main__":
    # 快速测试：打印数据信息
    train_loader, val_loader, names = build_loaders()
    print(f"\n类别名: {names}")
    print(f"训练 batch 数: {len(train_loader)}")
    print(f"验证 batch 数: {len(val_loader)}")

    # 测试一个 batch
    x, y = next(iter(train_loader))
    print(f"输入 shape: {x.shape}")   # 应为 [B, 3, 224, 224]
    print(f"标签 shape: {y.shape}")

    # 计算类别权重
    w = compute_class_weights()
    print(f"权重 tensor: {w}")
