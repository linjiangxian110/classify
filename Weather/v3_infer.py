import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Locate dinov3 source code
_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dinov3-main")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from dinov3.hub.backbones import dinov3_vitb16

# ==========================================
# 1. Path & Hyperparameter Config
# ==========================================
SPLIT_DATA_ROOT = "/mnt/JXYP/home/LYM/wether/split_dataset/"
CHECKPOINT_PATH = "./output_weathernet_finetune/finetuned_model_a100.pt"
IMG_SIZE = 384
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ==========================================
# 2. GeM Pooling Module Definition
# ==========================================
class GeMPooling(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1. / self.p)

# ==========================================
# 3. Load Test Dataset
# ==========================================
print(f"[*] Loading test dataset from: {SPLIT_DATA_ROOT}/test")
dataset = load_dataset("imagefolder", data_dir=os.path.join(SPLIT_DATA_ROOT, "test"))
test_ds = dataset["train"]

class_names = test_ds.features["label"].names
num_classes = len(class_names)
print(f"[*] Test set samples: {len(test_ds)} | Classes: {class_names}")

test_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(int(IMG_SIZE * 1.14), interpolation=v2.InterpolationMode.BICUBIC),
    v2.CenterCrop(IMG_SIZE),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def collate_fn(batch):
    images, labels = [], []
    for item in batch:
        img = item["image"].convert("RGB")
        images.append(img)
        labels.append(item["label"])
    return images, torch.tensor(labels, dtype=torch.long)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

# ==========================================
# 4. Initialize Model & Load Checkpoint
# ==========================================
print(f"[*] Loading checkpoint from: {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# Build backbone without pretrained weights for inference
backbone = dinov3_vitb16(pretrained=False)
embed_dim = backbone.embed_dim
hidden_dim = 2 * embed_dim

# Rebuild classification head
classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(hidden_dim, num_classes)
)
gem_pool = GeMPooling().to(DEVICE)

# Load saved weights
backbone.load_state_dict(ckpt["backbone_state_dict"])
classifier.load_state_dict(ckpt["classifier_state_dict"])

backbone.to(DEVICE)
classifier.to(DEVICE)

backbone.eval()
classifier.eval()

# ==========================================
# 5. Run Inference & Collect Predictions
# ==========================================
print("-" * 50)
print("Start evaluation on test set...")
all_preds = []
all_labels = []

with torch.no_grad():
    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            x_stack = torch.stack([test_transform(img) for img in imgs]).to(DEVICE)
            
            feat_out = backbone.forward_features(x_stack)
            cls_token = feat_out["x_norm_clstoken"]
            patch_gem = gem_pool(feat_out["x_norm_patchtokens"])
            feat = torch.cat([cls_token, patch_gem], dim=1)
            
            logits = classifier(feat)
            preds = logits.argmax(dim=1).cpu()
            
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

# ==========================================
# 6. Calculate Metrics & Print Report
# ==========================================
acc = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average='macro')

print("\n" + "=" * 50)
print(f"Test Set Accuracy : {acc * 100:.2f}%")
print(f"Test Set Macro F1 : {macro_f1:.4f}")
print("=" * 50)
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))