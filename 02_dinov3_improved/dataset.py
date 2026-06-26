"""
数据加载 — DINOv3 训练/验证，含 MixUp + RandAugment
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
from sklearn.model_selection import StratifiedShuffleSplit
from config import (
    DATA_DIR, IMG_SIZE, RESIZE_VAL, BATCH_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, VAL_RATIO, SEED,
    NUM_CLASSES, MIXUP_ALPHA,
)


def get_train_transforms():
    """训练增强：RandAugment + 随机裁剪"""
    return v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandAugment(num_ops=2, magnitude=5),  # 保守强度
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    """验证/推理预处理：Resize → CenterCrop → Normalize"""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(RESIZE_VAL, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(IMG_SIZE),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """
    MixUp 增强：两张随机混合。
    返回 (mixed_x, y_a, y_b, lam)
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp 损失"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def build_loaders(data_dir=None, batch_size=None):
    """构建 DataLoader"""
    data_dir = data_dir or DATA_DIR
    batch_size = batch_size or BATCH_SIZE

    # 两个版本的 dataset（训练增强 vs 验证预处理）
    train_full = datasets.ImageFolder(data_dir, transform=None)
    val_full = datasets.ImageFolder(data_dir, transform=None)
    class_names = train_full.classes

    total = len(train_full)
    n_val = int(total * VAL_RATIO)
    n_train = total - n_val

    # 分层划分
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=SEED)
    train_idx, val_idx = next(sss.split(np.zeros(total), train_full.targets))

    # 使用 Subset + 延迟 transform
    train_set = TransformSubset(train_full, train_idx, get_train_transforms())
    val_set = TransformSubset(val_full, val_idx, get_val_transforms())

    print(f"[Data] 训练: {n_train}  验证: {n_val}  类别: {class_names}")

    val_targets = [train_full.targets[i] for i in val_idx]
    val_dist = np.bincount(val_targets, minlength=NUM_CLASSES)
    print(f"[Data] 验证各类: {list(val_dist)}")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return train_loader, val_loader, class_names


class TransformSubset(torch.utils.data.Dataset):
    """Subset + 延迟 transform 包装器"""
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


def compute_class_weights(data_dir=None):
    data_dir = data_dir or DATA_DIR
    full_set = datasets.ImageFolder(data_dir)
    targets = np.array(full_set.targets)
    counts = np.bincount(targets, minlength=NUM_CLASSES)
    total = counts.sum()
    weights = total / (NUM_CLASSES * counts)
    w = torch.tensor(weights, dtype=torch.float32)
    print(f"[Data] 类别权重: {dict(zip(full_set.classes, w.tolist()))}")
    return w


if __name__ == "__main__":
    tl, vl, names = build_loaders()
    x, y = next(iter(tl))
    print(f"Batch shape: {x.shape}, labels: {y.shape}")
    w = compute_class_weights()
