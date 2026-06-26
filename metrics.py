"""
评估指标 — Macro F1 / per-class F1 / confusion matrix
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头模式，服务器不弹窗
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)


def compute_metrics(y_true, y_pred, label_names=None):
    """
    计算完整的评估指标。

    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        label_names: 类别名列表

    Returns:
        dict: {
            "macro_f1", "micro_f1", "accuracy",
            "per_class_f1": {label: f1},
            "per_class_precision": {label: precision},
            "per_class_recall": {label: recall},
            "confusion_matrix": np.ndarray,
        }
    """
    label_names = label_names or [str(i) for i in range(4)]

    results = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "accuracy": np.mean(np.array(y_true) == np.array(y_pred)),
        "per_class_f1": dict(zip(
            label_names,
            f1_score(y_true, y_pred, average=None, zero_division=0),
        )),
        "per_class_precision": dict(zip(
            label_names,
            precision_score(y_true, y_pred, average=None, zero_division=0),
        )),
        "per_class_recall": dict(zip(
            label_names,
            recall_score(y_true, y_pred, average=None, zero_division=0),
        )),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    return results


def print_metrics(results, label_names=None):
    """格式化打印评估结果"""
    label_names = label_names or list(results["per_class_f1"].keys())

    print(f"\n{'='*55}")
    print(f"  Macro F1:   {results['macro_f1']:.4f}")
    print(f"  Micro F1:   {results['micro_f1']:.4f}")
    print(f"  Accuracy:   {results['accuracy']:.4f}")
    print(f"{'='*55}")
    print(f"  {'类别':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*42}")
    for name in label_names:
        print(f"  {name:<12} "
              f"{results['per_class_precision'][name]:>10.4f} "
              f"{results['per_class_recall'][name]:>10.4f} "
              f"{results['per_class_f1'][name]:>10.4f}")
    print(f"{'='*55}")


def plot_confusion_matrix(cm, label_names, save_path="confusion_matrix.png"):
    """绘制并保存混淆矩阵"""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    # 标注数字
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center",
                    fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, fontsize=11)
    ax.set_yticklabels(label_names, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Metrics] 混淆矩阵已保存至: {save_path}")
