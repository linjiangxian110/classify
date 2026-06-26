"""
天气识别比赛 — 训练入口
支持 EfficientNet / ResNet / MobileNetV3 / ConvNeXt
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler as lr_scheduler_module

from config import (
    MODEL_NAME, NUM_CLASSES, PRETRAINED,
    EPOCHS, LR, LR_MIN, WARMUP_EPOCHS, BATCH_SIZE,
    USE_CLASS_WEIGHTS, DEVICE, LABELS,
    EARLY_STOP_PATIENCE, MONITOR_METRIC,
    LR_SCHEDULER, RESULT_DIR, SEED,
)
from model_zoo import create_model
from dataset import (
    build_loaders, compute_class_weights,
    get_val_transforms,
)
from metrics import compute_metrics, print_metrics, plot_confusion_matrix
from utils import (
    EarlyStopping, build_optimizer, save_checkpoint,
    save_final_model, AverageMeter, format_time,
)


def set_seed(seed=SEED):
    """固定随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, epoch):
    """训练一个 epoch"""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        # 统计
        pred = out.argmax(dim=1)
        correct = (pred == y).sum().item()

        loss_meter.update(loss.item(), x.size(0))
        acc_meter.update(correct / x.size(0), x.size(0))

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(model, loader, criterion):
    """验证一个 epoch — 返回完整指标"""
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)

        loss_meter.update(loss.item(), x.size(0))
        all_preds.extend(out.argmax(dim=1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, LABELS)
    metrics["val_loss"] = loss_meter.avg
    metrics["val_acc"] = metrics["accuracy"]  # 别名
    return metrics


def train():
    """主训练流程"""
    set_seed(SEED)
    print(f"\n{'='*55}")
    print(f"  天气识别训练 — {MODEL_NAME}")
    print(f"  设备: {DEVICE}")
    print(f"{'='*55}\n")

    # ── 1. 数据 ──
    train_loader, val_loader, class_names = build_loaders()
    print(f"[Data] 类别名: {class_names}\n")

    # ── 2. 模型 ──
    model = create_model(MODEL_NAME, NUM_CLASSES, PRETRAINED)
    model = model.to(DEVICE)

    # ── 3. Loss（类别权重处理不均衡）──
    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights().to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── 4. 优化器 + 调度器 ──
    optimizer = build_optimizer(model)
    scheduler = lr_scheduler_module.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR_MIN,
    )

    # ── 5. 早停 ──
    early_stopper = EarlyStopping(
        patience=EARLY_STOP_PATIENCE, mode="max",
    )

    # ── 6. 训练循环 ──
    best_f1 = 0.0
    best_epoch = 0
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch,
        )

        # 验证
        val_metrics = validate(model, val_loader, criterion)
        val_f1 = val_metrics["macro_f1"]

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # 判断是否最佳
        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1
            best_epoch = epoch

        # 打印
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"lr={current_lr:.2e} | "
            f"time={format_time(epoch_time)} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"val_acc={val_metrics['val_acc']:.4f} | "
            f"val_macro_f1={val_f1:.4f} {'★' if is_best else ''}"
        )

        # 每 5 轮 & 最佳轮次打印各类 F1
        if epoch % 5 == 0 or is_best:
            per_class = val_metrics["per_class_f1"]
            pfx = "  >> per-class F1: "
            print(pfx + " | ".join(
                f"{name}={per_class[name]:.4f}" for name in LABELS
            ))

        # 保存检查点
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            MODEL_NAME, is_best=is_best,
        )

        # 早停检查
        if early_stopper(epoch, val_f1):
            break

        print()  # 空行

    # ── 7. 训练完成 ──
    total_time = time.time() - train_start
    print(f"\n{'='*55}")
    print(f"  训练完成！总耗时: {format_time(total_time)}")
    print(f"  最优 Macro F1: {best_f1:.4f}  @ epoch {best_epoch}")
    print(f"{'='*55}")

    # ── 8. 加载最佳模型，画混淆矩阵 ──
    best_path = f"./checkpoints/{MODEL_NAME}_best.pth"
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Final] 已加载最佳模型: {best_path}")

    # 最终验证
    print("\n[Final] 最终验证集评估:")
    final_metrics = validate(model, val_loader, criterion)
    print_metrics(final_metrics)

    # 保存混淆矩阵
    os.makedirs(RESULT_DIR, exist_ok=True)
    cm_path = os.path.join(RESULT_DIR, f"{MODEL_NAME}_confusion_matrix.png")
    plot_confusion_matrix(
        final_metrics["confusion_matrix"], LABELS, cm_path,
    )

    # ── 9. 保存最终权重 ──
    weight_path = save_final_model(model, MODEL_NAME)
    print(f"[Final] 提交用权重: {weight_path}")
    print(f"[Final] 请将此文件放入 main.py 同目录的 results/ 文件夹\n")

    return model, final_metrics


if __name__ == "__main__":
    train()
