import torch
import torch.nn as nn
import timm
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from safetensors.torch import load_file

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN LƯU KẾT QUẢ
# ==========================================
OUTPUT_DIR = "src/results/final_comparison"  # <--- Sẽ lưu vào đây
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. MODEL DEFINITIONS
# ==========================================

class EfficientNetBinary(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=0)
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)

class ConvNextBinary(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=False, num_classes=0)
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Flatten(1),
            nn.Dropout(0.5),
            nn.Linear(in_features, 1)
        )
    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)

class ViTLoRA(nn.Module):
    def __init__(self, pretrained_path=None, lora_rank=16):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.backbone(x).squeeze(1)

# ==========================================
# 3. DATA UTILS
# ==========================================

class FaceLivenessDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        if os.path.exists(img_dir):
            files_set = set(os.listdir(img_dir))
            self.df = self.df[self.df['filepath'].apply(lambda x: os.path.basename(x) in files_set)].reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filepath'])
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224))
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(int(row['label']), dtype=torch.float)
        return img, label

def get_dataloader(config, input_size):
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_csv = os.path.join(config['dataset']['split_dir'], "test.csv")
    dataset = FaceLivenessDataset(test_csv, config['dataset']['img_dir'], transform)
    return DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# ==========================================
# 4. PLOTTING & SAVING
# ==========================================

def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_probs.extend(probs)
            y_true.extend(labels.numpy())
    return np.array(y_true), np.array(y_probs)

def plot_confusion_matrices(results, save_path):
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1: axes = [axes]
    
    for i, (name, metrics) in enumerate(results.items()):
        cm = metrics['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    xticklabels=['Live', 'Spoof'], yticklabels=['Live', 'Spoof'])
        axes[i].set_title(f"{name}\nAcc: {metrics['acc']:.4f}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"--> Đã lưu biểu đồ Confusion Matrix: {save_path}")

def plot_roc_curves(results, save_path):
    plt.figure(figsize=(10, 8))
    for name, metrics in results.items():
        fpr, tpr = metrics['fpr_curve'], metrics['tpr_curve']
        roc_auc = metrics['auc']
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('So sánh ROC Curve giữa các Model')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    print(f"--> Đã lưu biểu đồ ROC: {save_path}")

def save_report_to_txt(report_str, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"--> Đã lưu báo cáo chi tiết: {save_path}")

# ==========================================
# 5. MAIN
# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- BẮT ĐẦU ĐÁNH GIÁ TRÊN: {device} ---")
    
    # Load config để lấy đường dẫn data
    config_path = "config/efficientnet.yaml"
    if not os.path.exists(config_path):
        print("Không tìm thấy config/efficientnet.yaml")
        return
    with open(config_path) as f: config = yaml.safe_load(f)
    
    # Danh sách model cần đánh giá
    models_config = [
        {"name": "EfficientNet", "class": EfficientNetBinary, "path": "checkpoints/efficientnet/best.pt", "size": 260},
        {"name": "ConvNeXt",     "class": ConvNextBinary,     "path": "checkpoints/convnext/best.pt",     "size": 224},
        {"name": "ViT",          "class": ViTLoRA,            "path": "checkpoints/vit/best.pt",          "size": 224}
    ]
    
    results = {}
    report_lines = []
    
    # --- EVALUATION LOOP ---
    for cfg in models_config:
        print(f"\nDang xử lý: {cfg['name']}...")
        
        if not os.path.exists(cfg['path']):
            print(f"⚠️ CẢNH BÁO: Không tìm thấy file {cfg['path']}. Bỏ qua model này.")
            continue
            
        # Load Model
        model = cfg['class'](pretrained_path=None)
        model.load_state_dict(torch.load(cfg['path'], map_location=device))
        model.to(device)
        
        # Load Data
        loader = get_dataloader(config, cfg['size'])
        
        # Inference
        y_true, y_probs = evaluate_model(model, loader, device)
        y_pred = (y_probs >= 0.5).astype(int)
        
        # Calculate Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        # TPR @ 1% FPR
        idx_1fpr = np.abs(fpr - 0.01).argmin()
        tpr_1fpr = tpr[idx_1fpr]
        
        results[cfg['name']] = {
            'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': roc_auc,
            'cm': cm, 'fpr_curve': fpr, 'tpr_curve': tpr, 'tpr_1fpr': tpr_1fpr
        }

    # --- TẠO BÁO CÁO ---
    header = f"{'MODEL':<15} | {'ACC':<8} | {'PRECISION':<10} | {'RECALL':<8} | {'F1':<8} | {'AUC':<8} | {'TPR@1%FPR':<10}"
    div = "-" * 90
    
    print("\n" + div)
    print(header)
    print(div)
    report_lines.append(div)
    report_lines.append(header)
    report_lines.append(div)
    
    for name, m in results.items():
        line = f"{name:<15} | {m['acc']:.4f}   | {m['prec']:.4f}     | {m['rec']:.4f}   | {m['f1']:.4f}   | {m['auc']:.4f}   | {m['tpr_1fpr']:.4f}"
        print(line)
        report_lines.append(line)
    
    print(div)
    report_lines.append(div)
    
    # Chi tiết Confusion Matrix
    report_lines.append("\nCHI TIẾT CONFUSION MATRIX:")
    for name, m in results.items():
        tn, fp, fn, tp = m['cm'].ravel()
        cm_text = f"\n[{name}]\n      Pred Live | Pred Spoof\nLive: {tn:^9} | {fp:^10}\nSpoof:{fn:^9} | {tp:^10}"
        print(cm_text)
        report_lines.append(cm_text)

    # --- LƯU FILE ---
    # 1. Lưu biểu đồ ROC
    plot_roc_curves(results, save_path=os.path.join(OUTPUT_DIR, "comparison_roc.png"))
    
    # 2. Lưu biểu đồ Confusion Matrix
    plot_confusion_matrices(results, save_path=os.path.join(OUTPUT_DIR, "comparison_cm.png"))
    
    # 3. Lưu báo cáo Text
    save_report_to_txt("\n".join(report_lines), os.path.join(OUTPUT_DIR, "final_report.txt"))

    print(f"\n HOÀN TẤT! Kiểm tra kết quả tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()