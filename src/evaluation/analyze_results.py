import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Import Model Classes
from src.models.build_model.convnext import ConvNextBinary
from src.models.build_model.efficientnet import EfficientNetBinary
from src.data.data_loader import build_dataloaders

# ================= CẤU HÌNH =================
OUTPUT_DIR = "src/results/final_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Định nghĩa đường dẫn config cho từng model
MODELS_CONFIG = {
    "ConvNeXt": {
        "log_file": "src/results/convnext/training_log.csv",
        "weights": "saved_models/convnext/best.pt",
        "model_class": ConvNextBinary,
        "input_size": 224
    },
    "EfficientNet": {
        "log_file": "src/results/efficientnet/training_log.csv",
        "weights": "checkpoints/efficientnet/best.pt",
        "model_class": EfficientNetBinary,
        "input_size": 260
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_learning_curves(model_name, log_path):
    print(f"[{model_name}] Dang ve bieu do Loss/Acc...")
    if not os.path.exists(log_path):
        print(f" -> [SKIP] Khong tim thay file log: {log_path}")
        return

    try:
        df = pd.read_csv(log_path)
        # Vẽ 2 biểu đồ cạnh nhau
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 1. Loss Curve
        # Kiểm tra tên cột trong CSV (phòng trường hợp khác nhau)
        cols = df.columns
        t_loss = df['Train_Loss'] if 'Train_Loss' in cols else df['T_Loss']
        v_loss = df['Val_Loss'] if 'Val_Loss' in cols else df['V_Loss']
        t_acc = df['Train_Acc'] if 'Train_Acc' in cols else df['T_Acc']
        v_acc = df['Val_Acc'] if 'Val_Acc' in cols else df['V_Acc']
        epochs = df['Epoch']

        ax1.plot(epochs, t_loss, label='Train Loss', color='blue', linewidth=2)
        ax1.plot(epochs, v_loss, label='Val Loss', color='red', linewidth=2, linestyle='--')
        ax1.set_title(f'{model_name}: Learning Curve (Loss)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Accuracy Curve
        ax2.plot(epochs, t_acc, label='Train Acc', color='green', linewidth=2)
        ax2.plot(epochs, v_acc, label='Val Acc', color='orange', linewidth=2, linestyle='--')
        ax2.set_title(f'{model_name}: Accuracy Evolution')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"{model_name}_learning_curve.png")
        plt.savefig(save_path)
        print(f" -> Da luu: {save_path}")
        plt.close()
    except Exception as e:
        print(f" -> Loi ve bieu do: {e}")

def plot_confusion_matrix(model_name, config):
    print(f"[{model_name}] Dang chay lai Test de ve Heatmap...")
    
    if not os.path.exists(config['weights']):
        print(f" -> [SKIP] Khong tim thay file weights: {config['weights']}")
        return

    try:
        # 1. Load Model
        model = config['model_class'](pretrained=False).to(device)
        state_dict = torch.load(config['weights'], map_location=device)
        
        # Fix lỗi nếu train bằng DataParallel (có prefix 'module.')
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        model.eval()

        # 2. Load Data Test
        loaders = build_dataloaders(
            img_dir="data/data_split", 
            input_size=config['input_size'], 
            batch_size=32, 
            num_workers=4
        )
        test_loader = loaders['test']

        # 3. Inference
        y_true = []
        y_pred = []
        
        print(" -> Dang du doan...")
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) >= 0.5).long().cpu().numpy()
                
                y_true.extend(labels.numpy())
                y_pred.extend(preds)

        # 4. Ve Heatmap
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Live', 'Spoof'], yticklabels=['Live', 'Spoof'],
                    annot_kws={"size": 16})
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'{model_name} Confusion Matrix', fontsize=15)
        
        save_path = os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png")
        plt.savefig(save_path)
        print(f" -> Da luu Heatmap: {save_path}")
        plt.close()
        
        # In ra terminal
        print("\n" + "="*40)
        print(f"REPORT CHI TIET: {model_name}")
        print(classification_report(y_true, y_pred, target_names=['Live', 'Spoof'], digits=4))
        print("="*40 + "\n")
        
    except Exception as e:
        print(f" -> Loi khi danh gia model: {e}")

def main():
    print("BAT DAU PHAN TICH KET QUA...")
    for name, config in MODELS_CONFIG.items():
        print(f"\n--- {name} ---")
        # 1. Ve bieu do Loss (Check Overfit)
        plot_learning_curves(name, config['log_file'])
        
        # 2. Ve Heatmap (Check nham lan)
        plot_confusion_matrix(name, config)
        
    print(f"Hoan tat! Kiem tra folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
