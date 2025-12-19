import os
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# Setup path
import sys
sys.path.append(os.getcwd())
from src.models.build_model.convnext import ConvNextBinary
from src.models.build_model.efficientnet import EfficientNetBinary
from src.models.build_model.vit import ViTBinary

# ================= CAU HINH =================
TEST_DIR = "data/data_split/test"
OUTPUT_DIR = "src/results/paper_figures_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS_CONFIG = {
    "ConvNeXt": {"class": ConvNextBinary, "path": "saved_models/convnext/best.pt", "size": 224},
    "EfficientNet": {"class": EfficientNetBinary, "path": "checkpoints/efficientnet/best.pt", "size": 260},
    "ViT": {"class": ViTBinary, "path": "checkpoints/vit/best.pt", "size": 224}
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(name, cfg):
    model = cfg['class'](pretrained=False).to(device)
    state = torch.load(cfg['path'], map_location=device)
    # Fix DataParallel prefix
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
    model.eval()
    return model

def get_spoof_type(filename):
    # Parse loai spoof tu ten file de hien thi cho "ngau" trong paper
    parts = os.path.splitext(filename)[0].split('_')
    return parts[-1] if len(parts) > 0 else "Spoof"

def main():
    # 1. Chon mau test (6 Real, 6 Spoof mix cac loai)
    live_dir = os.path.join(TEST_DIR, "0_live")
    spoof_dir = os.path.join(TEST_DIR, "1_spoof")
    
    live_files = [os.path.join(live_dir, f) for f in os.listdir(live_dir) if f.endswith(('.jpg', '.png'))]
    spoof_files = [os.path.join(spoof_path, f) for spoof_path in [spoof_dir] for f in os.listdir(spoof_path) if f.endswith(('.jpg', '.png'))]
    
    selected_samples = []
    # Lay 6 anh Real
    selected_samples.extend([(f, 0, "Real") for f in random.sample(live_files, 6)])
    # Lay 6 anh Spoof (co gang lay da dang loai)
    selected_samples.extend([(f, 1, get_spoof_type(f)) for f in random.sample(spoof_files, 6)])
    
    random.shuffle(selected_samples)

    # 2. Chay cho tung Model
    for model_name, cfg in MODELS_CONFIG.items():
        print(f"Generating visual results for {model_name}...")
        model = load_model(model_name, cfg)
        size = cfg['size']
        tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Ve Grid 3x4
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        
        for i, (img_path, gt_label, spoof_type) in enumerate(selected_samples):
            ax = axes[i//4, i%4]
            img_pil = Image.open(img_path).convert('RGB')
            
            # Inference
            input_tensor = tf(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_tensor)
                prob = torch.sigmoid(out).item()
            
            pred_label = 1 if prob >= 0.5 else 0
            conf = prob if pred_label == 1 else 1 - prob
            
            # Chuan bi text
            gt_text = f"GT: {spoof_type.upper()}"
            pred_text = "REAL" if pred_label == 0 else "SPOOF"
            color = "green" if pred_label == gt_label else "red"
            
            ax.imshow(img_pil)
            ax.set_title(f"{gt_text}\nPred: {pred_text} ({conf:.1%})", 
                         color=color, fontsize=11, fontweight='bold')
            ax.axis('off')
            
            # Them vien khung de bat mat
            rect = plt.Rectangle((0,0), 1, 1, transform=ax.transAxes, color=color, fill=False, lw=3)
            ax.add_patch(rect)

        plt.suptitle(f"Qualitative Results: {model_name} Prediction on Test Samples", fontsize=20, y=0.98)
        save_path = os.path.join(OUTPUT_DIR, f"Fig8_Qualitative_{model_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" -> Saved: {save_path}")

if __name__ == "__main__":
    main()
