"""
DINOv3 ViT-B/16 + GeM Pooling + CLS Concat 分类头
改进：DropPath 正则化 + EMA
"""
import copy
import torch
import torch.nn as nn
from dinov3.hub.backbones import dinov3_vitb16
from config import (
    NUM_CLASSES, EMBED_DIM, HIDDEN_DIM,
    DROPOUT, DROP_PATH_RATE,
)


def _set_drop_path(model, drop_rate):
    """遍历 ViT blocks，设置 stochastic depth 概率"""
    for module in model.modules():
        if hasattr(module, "sample_drop_ratio"):
            module.sample_drop_ratio = drop_rate


class GeMPool(nn.Module):
    """Generalized Mean Pooling (p=3)"""
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)


class DINOv3Classifier(nn.Module):
    """
    DINOv3 ViT-B/16 + CLS+GeM 天气分类模型。
    前向返回 logits。
    """

    def __init__(self, pretrained_path=None, drop_path_rate=DROP_PATH_RATE):
        super().__init__()

        # ── Backbone ──
        self.backbone = dinov3_vitb16(pretrained=False)
        if drop_path_rate > 0:
            _set_drop_path(self.backbone, drop_path_rate)

        # ── GeM + 分类头 ──
        self.gem_pool = GeMPool()
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

        # ── 加载权重 ──
        if pretrained_path:
            print(f"[Model] 加载权重: {pretrained_path}")
            state = torch.load(pretrained_path, map_location="cpu", weights_only=False)

            # 微调过的 checkpoint（含 backbone + classifier）
            if "backbone_state_dict" in state:
                print("[Model] 检测到微调 checkoint → 加载 backbone + classifier")
                self.backbone.load_state_dict(state["backbone_state_dict"])
                self.classifier.load_state_dict(state["classifier_state_dict"])
                if "gem_pool_state_dict" in state:
                    self.gem_pool.load_state_dict(state["gem_pool_state_dict"])
            # 纯预训练权重
            elif "teacher" in state:
                print("[Model] 检测到 teacher 格式 → 只加载 backbone")
                self.backbone.load_state_dict(state["teacher"], strict=False)
            else:
                print("[Model] 检测到裸 state_dict → 只加载 backbone")
                self.backbone.load_state_dict(state, strict=False)

        # ── 统计 ──
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] 总参数: {total/1e6:.1f}M  可训练: {trainable/1e6:.1f}M")
        print(f"[Model] DropPath={drop_path_rate}  Dropout={DROPOUT}")

    def forward(self, x):
        feat = self.backbone.forward_features(x)
        cls_token = feat["x_norm_clstoken"]            # [B, 768]
        patch_gem = self.gem_pool(feat["x_norm_patchtokens"])  # [B, 768]
        fused = torch.cat([cls_token, patch_gem], dim=1)       # [B, 1536]
        return self.classifier(fused)


class EMAModel:
    """
    指数移动平均包装器。
    用法:
        ema = EMAModel(model, decay=0.999)
        for step in ...:
            ...训练...
            ema.update()
        ema.apply()  # 推理前执行
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self._backup = {}

    def update(self):
        with torch.no_grad():
            for k, param in self.model.state_dict().items():
                if param.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(
                        param.detach(), alpha=1 - self.decay
                    )
                else:
                    self.shadow[k].copy_(param.detach())

    def apply(self):
        """用 EMA 权重替换模型权重"""
        self._backup = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow)

    def restore(self):
        """恢复原始权重"""
        self.model.load_state_dict(self._backup)
