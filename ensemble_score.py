import torch
import yaml
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

# Import Model
from src.models.build_model.efficientnet import EfficientNetBinary
from src.models.build_model.vit import ViTLoRA

# --- CONFIG ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EFF_PATH = "checkpoints/efficientnet/best.pt"
VIT_PATH = "checkpoints/vit/best.pt"

# --- DATASET ---
class FaceLivenessDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        if os.path.exists(img_dir):
            files_set = set(os.listdir(img_dir))
            self.df = self.df[self.df['filepath'].apply(lambda x: os.path.basename(x) in files_set)].reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['filepath'])).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, torch.tensor(int(row['label']), dtype=torch.float)

def get_loader(input_size):
    # Load config để lấy đường dẫn
    with open("config/efficientnet.yaml") as f: config = yaml.safe_load(f)
    transform = T.Compose([
        T.Resize((input_size, input_size)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = FaceLivenessDataset(
        os.path.join(config['dataset']['split_dir'], "test.csv"),
        config['dataset']['img_dir'], transform
    )
    return DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

def main():
    print("--- ĐANG CHẠY ENSEMBLE (EFFICIENTNET + VIT) ---")
    
    # 1. Load EfficientNet (Size 260)
    print("1. Loading EfficientNet...")
    model_eff = EfficientNetBinary(pretrained_path=None).to(device)
    model_eff.load_state_dict(torch.load(EFF_PATH, map_location=device))
    model_eff.eval()
    loader_eff = get_loader(260)
    
    # 2. Load ViT (Size 224)
    print("2. Loading ViT...")
    model_vit = ViTLoRA(pretrained_path=None).to(device)
    model_vit.load_state_dict(torch.load(VIT_PATH, map_location=device))
    model_vit.eval()
    loader_vit = get_loader(224) # Lưu ý ViT dùng size 224
    
    # 3. Inference
    y_true = []
    probs_eff = []
    probs_vit = []
    
    print("3. Running Inference EfficientNet...")
    with torch.no_grad():
        for imgs, labels in loader_eff:
            out = model_eff(imgs.to(device))
            probs_eff.extend(torch.sigmoid(out).cpu().numpy())
            y_true.extend(labels.numpy())
            
    print("4. Running Inference ViT...")
    with torch.no_grad():
        for imgs, _ in loader_vit:
            out = model_vit(imgs.to(device))
            probs_vit.extend(torch.sigmoid(out).cpu().numpy())
            
    # 4. Combine (Trọng số 0.6 cho Eff, 0.4 cho ViT vì Eff ngon hơn)
    probs_eff = np.array(probs_eff)
    probs_vit = np.array(probs_vit)
    y_true = np.array(y_true)
    
    # Công thức Ensemble
    final_probs = (0.7 * probs_eff) + (0.3 * probs_vit)
    y_pred = (final_probs >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    print("\n" + "="*40)
    print(f"KẾT QUẢ ENSEMBLE (0.7*Eff + 0.3*ViT)")
    print("="*40)
    print(f"Accuracy: {acc:.4f}")
    print(f"CM: [TN:{tn} FP:{fp}] [FN:{fn} TP:{tp}]")
    
    if acc > 0.9370:
        print("🚀 TUYỆT VỜI! Ensemble đã cải thiện kết quả!")
    else:
        print("😐 Kết quả không tăng. Hãy dùng EfficientNet đơn lẻ.")

if __name__ == "__main__":
    main()