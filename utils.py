"""
工具函数 — 早停、学习率调度、模型保存、日志
"""
import os
import time
import torch
import torch.optim as optim
from config import (
    LR, LR_MIN, WEIGHT_DECAY, ADAMW_BETAS, ADAMW_EPS,
    WARMUP_EPOCHS, EPOCHS, OPTIMIZER,
    EARLY_STOP_PATIENCE, MONITOR_METRIC, SAVE_BEST_ONLY,
    CHECKPOINT_DIR, RESULT_DIR,
)


class EarlyStopping:
    """早停：监控指标连续 N 轮不涨就停"""

    def __init__(self, patience=10, mode="max", verbose=True):
        """
        Args:
            patience: 容忍轮数
            mode: "max" 指标越大越好, "min" 指标越小越好
            verbose: 是否打印信息
        """
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == "max":
            self.monitor_op = lambda current, best: current > best
            self.delta_sign = 1
        else:
            self.monitor_op = lambda current, best: current < best
            self.delta_sign = -1

    def __call__(self, epoch, score):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStop] 指标未提升 ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"[EarlyStop] 停止训练，最优轮次: {self.best_epoch}, "
                      f"最优分数: {self.best_score:.4f}")

        return self.early_stop


def build_optimizer(model):
    """构建优化器"""
    if OPTIMIZER.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=LR,
            betas=ADAMW_BETAS,
            eps=ADAMW_EPS,
            weight_decay=WEIGHT_DECAY,
        )
    elif OPTIMIZER.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
    elif OPTIMIZER.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=LR,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"不支持的优化器: {OPTIMIZER}")


def build_scheduler(optimizer, steps_per_epoch):
    """构建学习率调度器（cosine + warmup）"""
    # cosine 退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS * steps_per_epoch,
        eta_min=LR_MIN,
    )
    return scheduler
    # 注：warmup 在训练循环中手动实现


def save_checkpoint(model, optimizer, epoch, metrics, model_name,
                    is_best=False):
    """
    保存检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        metrics: 指标字典
        model_name: 模型名
        is_best: 是否为最佳模型
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    # 每轮保存（可选）
    if not SAVE_BEST_ONLY:
        path = os.path.join(CHECKPOINT_DIR, f"{model_name}_epoch{epoch}.pth")
        torch.save(ckpt, path)

    # 最佳模型覆盖保存
    if is_best:
        path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
        torch.save(ckpt, path)
        print(f"[Checkpoint] 最佳模型已保存: {path}")


def save_final_model(model, model_name):
    """保存最终提交用的权重文件（只含 state_dict）"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"{model_name}_final.pth")
    torch.save(model.state_dict(), path)
    print(f"[Result] 最终权重已保存: {path}")
    return path


class AverageMeter:
    """平均值追踪器"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(seconds):
    """格式化时间为 mm:ss"""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
