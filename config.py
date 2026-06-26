"""
天气识别比赛 — 全局配置文件
改超参数只改这一个文件，训练和推理自动同步
"""
import os

# ============================================================
# 数据路径（本地和服务器各配各的，改这里就行）
# ============================================================
# 本地路径（Windows）
DATA_DIR_LOCAL = r"D:\Temporary\文档\智海调优\天气识别\train"
# 服务器路径（Linux），训练时用这个
DATA_DIR_SERVER = "/home/jovyan/work/datasets/6a39ed934d7b489daf5f80a4-momodel/train"

# 自动检测：如果本地路径存在就用本地，否则用服务器路径
DATA_DIR = DATA_DIR_LOCAL if os.path.exists(DATA_DIR_LOCAL) else DATA_DIR_SERVER

# 输出目录
CHECKPOINT_DIR = "./checkpoints"
RESULT_DIR = "./results"
LOG_DIR = "./logs"

# ============================================================
# 模型配置
# ============================================================
# 可选: efficientnet_b0 / resnet18 / mobilenetv3_large_100 / convnext_tiny
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 4
PRETRAINED = True          # 使用 ImageNet 预训练权重
FREEZE_BACKBONE = False    # False = 全模型微调；True = 只训分类头

# ============================================================
# 类别映射（必须与数据子文件夹名一致）
# ============================================================
LABELS = ["cloudy", "rainy", "snowy", "sunny"]
LABEL2ID = {name: idx for idx, name in enumerate(LABELS)}
ID2LABEL = {idx: name for idx, name in enumerate(LABELS)}

# ============================================================
# 图像预处理
# ============================================================
IMG_SIZE = 224              # 最终输入尺寸
RESIZE_VAL = 256            # 验证/推理时的 resize 尺寸

# ImageNet 标准均值和标准差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================
# 训练超参数
# ============================================================
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
LR_MIN = 1e-6               # cosine scheduler 最低学习率
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2           # 预热轮数

# 验证集比例
VAL_RATIO = 0.15
# 随机种子
SEED = 42

# ============================================================
# 类别权重（处理不均衡，自动计算或手动指定）
# ============================================================
# True = 根据训练集各类数量自动计算 class weights
# False = 使用下方手动指定的 CLASS_WEIGHTS_MANUAL
USE_CLASS_WEIGHTS = True
CLASS_WEIGHTS_MANUAL = [1.0, 2.5, 2.5, 1.0]  # rainy/snowy 加权

# ============================================================
# 优化器与调度器
# ============================================================
OPTIMIZER = "adamw"          # adamw / adam / sgd
# AdamW 默认参数
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8

# 学习率调度策略: cosine / step / plateau
LR_SCHEDULER = "cosine"

# ============================================================
# 早停与模型保存
# ============================================================
EARLY_STOP_PATIENCE = 10     # 验证 F1 连续 N 轮不涨就停
MONITOR_METRIC = "val_macro_f1"  # 监控指标
SAVE_BEST_ONLY = True        # 只保存最优模型

# ============================================================
# 设备
# ============================================================
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 打印当前配置
# ============================================================
if __name__ == "__main__":
    print(f"MODEL:        {MODEL_NAME}")
    print(f"DATA_DIR:     {DATA_DIR}")
    print(f"DEVICE:       {DEVICE}")
    print(f"IMG_SIZE:     {IMG_SIZE}")
    print(f"BATCH_SIZE:   {BATCH_SIZE}")
    print(f"EPOCHS:       {EPOCHS}")
    print(f"LR:           {LR}")
    print(f"NUM_CLASSES:  {NUM_CLASSES}")
    print(f"PRETRAINED:   {PRETRAINED}")
