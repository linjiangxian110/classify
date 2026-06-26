"""
DINOv3 ViT-B/16 改进训练脚本
改进点：EMA / MixUp / Label Smoothing / DropPath / CosineWarmup
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# dinov3 源码路径
_DINOV3 = os.path.join(os.path.dirname(__file__), "..", "Weather", "dinov3-main")
if _DINOV3 not in sys.path:
    sys.path.insert(0, _DINOV3)

from config import *
from model import DINOv3Classifier, EMAModel
from dataset import (
    build_loaders, mixup_data, mixup_criterion,
    get_val_transforms,
)
from sklearn.metrics import f1_score


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


# ============================================================
# 训练/验证
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler=None):
    model.train()
    loss_sum = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        # MixUp
        if MIXUP_ALPHA > 0:
            x, y_a, y_b, lam = mixup_data(x, y, MIXUP_ALPHA)
        else:
            y_a, y_b, lam = y, y, 1.0

        optimizer.zero_grad()

        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * x.size(0)
        n += x.size(0)

    return loss_sum / n


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    loss_sum, n = 0.0, 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)

        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    return {
        "val_loss": loss_sum / n,
        "val_acc": np.mean(np.array(all_preds) == np.array(all_labels)),
        "val_macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "y_true": all_labels,
        "y_pred": all_preds,
    }


# ============================================================
# 主函数
# ============================================================
def train():
    set_seed()

    print(f"\n{'='*55}")
    print(f"  DINOv3 ViT-B/16 改进训练")
    print(f"  GPU: {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'}")
    print(f"  MixUp={MIXUP_ALPHA}  EMA={EMA_DECAY}  DropPath={DROP_PATH_RATE}")
    print(f"  LabelSmooth={LABEL_SMOOTHING}")
    print(f"{'='*55}\n")

    # ── 数据 ──
    if VAL_RATIO <= 0:
        # 全量训练，无验证集
        print("[Data] 全量训练模式（VAL_RATIO=0）")
        from torch.utils.data import DataLoader
        from torchvision import datasets
        from dataset import get_train_transforms
        full_set = datasets.ImageFolder(DATA_DIR, transform=get_train_transforms())
        train_loader = DataLoader(full_set, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader = None
        class_names = full_set.classes
        print(f"[Data] 全量: {len(full_set)} 张")
    else:
        train_loader, val_loader, class_names = build_loaders()

    # ── 模型 ──
    model = DINOv3Classifier(
        pretrained_path=PRETRAIN_PATH,
        drop_path_rate=DROP_PATH_RATE,
    ).to(DEVICE)

    # ── EMA ──
    ema = EMAModel(model, decay=EMA_DECAY) if EMA_DECAY > 0 else None

    # ── Loss ──
    criterion = nn.CrossEntropyLoss()

    # ── 优化器 + 调度器 ──
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = None  # 不用 AMP，保持简单
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR_MIN,
    )

    # ── 训练循环 ──
    best_f1 = 0.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # LR warmup
        if epoch <= WARMUP_EPOCHS:
            lr = LR * epoch / WARMUP_EPOCHS
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)

        if epoch > WARMUP_EPOCHS:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # EMA 更新
        if ema:
            ema.update()

        # 验证（如果有验证集）
        if val_loader is not None:
            metrics = validate(model, val_loader, criterion)
            val_f1 = metrics["val_macro_f1"]
            is_best = val_f1 > best_f1
            if is_best:
                best_f1 = val_f1
                best_epoch = epoch
                no_improve = 0
                save_checkpoint(model, epoch, metrics, is_best=True)
            else:
                no_improve += 1
        else:
            # 全量训练：只看 train_loss，每轮都存
            val_f1 = 0.0
            is_best = True
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(model, epoch, {"val_macro_f1": 0.0}, is_best=True)

        elapsed = time.time() - t0
        tag = "val_f1" if val_loader else "full"
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"lr={current_lr:.2e} | "
            f"time={elapsed//60:02.0f}:{elapsed%60:02.0f} | "
            f"train_loss={train_loss:.4f} | "
            f"{tag}={val_f1:.4f}"
            f"{' ★' if is_best else ''}"
        )

        if val_loader and no_improve >= EARLY_STOP_PATIENCE:
            print(f"[EarlyStop] 停止于 epoch {epoch}")
            break

    print(f"\n最佳 Macro F1: {best_f1:.4f} @ epoch {best_epoch}")

    # ── 最终保存 ──
    if val_loader is not None and ema:
        ema.apply()
        print("\n[EMA] 最终验证:")
        final_m = validate(model, val_loader, criterion)
        print(f"  Macro F1: {final_m['val_macro_f1']:.4f}  Acc: {final_m['val_acc']:.4f}")
        save_final(model, suffix="_ema")
        ema.restore()

    save_final(model, suffix="")
    return model


# ============================================================
# 保存
# ============================================================
def save_checkpoint(model, epoch, metrics, is_best=False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "backbone_state_dict": model.backbone.state_dict(),
        "classifier_state_dict": model.classifier.state_dict(),
        "gem_pool_state_dict": model.gem_pool.state_dict(),
        "macro_f1": metrics["val_macro_f1"],
    }
    path = os.path.join(CHECKPOINT_DIR, f"dinov3_{EXP_NAME}.pth")
    torch.save(ckpt, path)
    if is_best:
        print(f"[Save] {path}  (F1={metrics['val_macro_f1']:.4f})")


def save_final(model, suffix=""):
    os.makedirs(RESULT_DIR, exist_ok=True)
    name = f"dinov3_{EXP_NAME}{suffix}.pt"
    path = os.path.join(RESULT_DIR, name)
    ckpt = {
        "backbone_state_dict": model.backbone.state_dict(),
        "classifier_state_dict": model.classifier.state_dict(),
        "gem_pool_state_dict": model.gem_pool.state_dict(),
    }
    torch.save(ckpt, path)
    print(f"[Save] {path}")


if __name__ == "__main__":
    train()
