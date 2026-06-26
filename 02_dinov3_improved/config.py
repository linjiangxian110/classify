"""
DINOv3 ViT-B/16 改进训练 — 配置文件
"""
import os
import sys
import torch

# ============================================================
# 路径
# ============================================================
DATA_DIR = "/mnt/data/lck/code/classify/train"
PRETRAIN_PATH = os.environ.get("WEATHER_PRETRAIN",
    "/mnt/data/lck/code/classify/finetuned_model_a100_full.pt")
EXP_NAME = os.environ.get("WEATHER_EXP_NAME", "default")
CHECKPOINT_DIR = f"./checkpoints/{EXP_NAME}"
RESULT_DIR = f"./results/{EXP_NAME}"

# dinov3 源码路径（相对于本文件）
_DINOV3_SRC = os.path.join(os.path.dirname(__file__), "..", "Weather", "dinov3-main")
if _DINOV3_SRC not in sys.path:
    sys.path.insert(0, _DINOV3_SRC)

# ============================================================
# 模型
# ============================================================
IMG_SIZE = 384
RESIZE_VAL = int(IMG_SIZE * 1.14)   # 438 → CenterCrop → 384
NUM_CLASSES = 4
LABELS = ["cloudy", "rainy", "snowy", "sunny"]
EMBED_DIM = 768
HIDDEN_DIM = 2 * EMBED_DIM          # CLS + GeM concat → 1536

# ============================================================
# 训练
# ============================================================
BATCH_SIZE = 16                      # ViT + 384px 显存占用大
EPOCHS = 20                        # 数据扩充后多训几轮
LR = 1e-4                          # 微调用低 LR
LR_MIN = 1e-6
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 2
VAL_RATIO = float(os.environ.get("WEATHER_VAL_RATIO", "0.15"))
SEED = 42
EARLY_STOP_PATIENCE = 7             # 20 轮微调，7 轮不涨就停

# ============================================================
# 改进：正则化
# ============================================================
LABEL_SMOOTHING = 0.0               # 先关闭，后续逐步加
DROPOUT = 0.2                       # 分类头 dropout
DROP_PATH_RATE = 0.1                # Step 2：加 stochastic depth
MIXUP_ALPHA = 0.0                   # 先关闭（天气分类对混合敏感）
EMA_DECAY = 0.0                     # TTA 无效，EMA 持平，全关

# ============================================================
# 图像预处理
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ============================================================
# 优化器
# ============================================================
OPTIMIZER = "adamw"
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8

# ============================================================
# 早停
# ============================================================
EARLY_STOP_PATIENCE = 7
MONITOR_METRIC = "val_macro_f1"

# ============================================================
# 设备
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 环境变量覆盖
MODEL_NAME = os.environ.get("WEATHER_MODEL", "dinov3_vitb16")
EPOCHS = int(os.environ.get("WEATHER_EPOCHS", str(EPOCHS)))
BATCH_SIZE = int(os.environ.get("WEATHER_BATCH", str(BATCH_SIZE)))
LR = float(os.environ.get("WEATHER_LR", str(LR)))

if __name__ == "__main__":
    print(f"MODEL:        {MODEL_NAME}")
    print(f"DATA_DIR:     {DATA_DIR}")
    print(f"DEVICE:       {DEVICE}")
    print(f"IMG_SIZE:     {IMG_SIZE}")
    print(f"BATCH:        {BATCH_SIZE}")
    print(f"EPOCHS:       {EPOCHS}")
    print(f"LR:           {LR}")
    print(f"MIXUP:        {MIXUP_ALPHA}")
    print(f"EMA:          {EMA_DECAY}")
    print(f"DROP_PATH:    {DROP_PATH_RATE}")
