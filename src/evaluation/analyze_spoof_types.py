import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import Models
from src.models.build_model.convnext import ConvNextBinary
from src.models.build_model.efficientnet import EfficientNetBinary
from src.models.build_model.vit import ViTBinary

# ================= CẤU HÌNH =================
TEST_SPOOF_DIR = "data/data_split/test/1_spoof"
OUTPUT_DIR = "src/results/final_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Định nghĩa Model và Weights
MODELS = {
    "ConvNeXt": {
        "class": ConvNextBinary,
        "weights": "saved_models/convnext/best.pt",
        "size": 224
    },
    "EfficientNet": {
        "class": EfficientNetBinary,
        "weights": "checkpoints/efficientnet/best.pt",
        "size": 260
    },
    "ViT": {
        "class": ViTBinary,
        "weights": "checkpoints/vit/best.pt",
        "size": 224
    }
}

# Định nghĩa các nhóm tấn công
SPOOF_CATEGORIES = {
    'Deepfake': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter', 'CelebA'],
    'Mask': ['silicon', 'mask', 'latex', 'UpperBodyMask', 'RegionMask', '3D_Mask'],
    'Print': ['Poster', 'Photo', 'A4', 'Print'],
    'Replay': ['Phone', 'PC', 'Pad', 'Screen', 'Replay']
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_spoof_category(filename):
    """Phân loại dựa trên tên file"""
    try:
        # Filename: Dataset_ID_1_SpoofType.jpg
        name_body = os.path.splitext(filename)[0]
        parts = name_body.split('_')
        raw_type = parts[-1]
        
        for cat, keywords in SPOOF_CATEGORIES.items():
            for kw in keywords:
                if kw.lower() in raw_type.lower():
                    return cat
        return "Other" # Các loại lạ hoặc chưa định nghĩa
    except:
        return "Unknown"

class SpoofTypeDataset(Dataset):
    def __init__(self, root_dir, input_size):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.png'))]
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        category = get_spoof_category(filename)
        img_path = os.path.join(self.root_dir, filename)
        
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except:
            img = torch.zeros((3, 224, 224)) # Dummy nếu lỗi ảnh
            
        return img, category

def load_model_instance(config):
    if not os.path.exists(config['weights']):
        return None
    model = config['class'](pretrained=False).to(device)
    state_dict = torch.load(config['weights'], map_location=device)
    
    # Fix keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def main():
    print("--- BAT DAU PHAN TICH CHI TIET LOAI SPOOF ---")
    
    if not os.path.exists(TEST_SPOOF_DIR):
        print(f"Loi: Khong tim thay thu muc {TEST_SPOOF_DIR}")
        return

    # Dataframe để lưu kết quả tổng hợp
    # Cấu trúc: Index=Spoof_Type, Columns=[Total_Images, ConvNeXt_Fail, EffNet_Fail...]
    results = {} 

    for model_name, config in MODELS.items():
        print(f"\n>> Dang danh gia model: {model_name}...")
        
        model = load_model_instance(config)
        if model is None:
            print(f"   [SKIP] Khong tim thay weights.")
            continue

        # Load Data
        dataset = SpoofTypeDataset(TEST_SPOOF_DIR, config['size'])
        loader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)
        
        # Thống kê tạm thời cho model này
        stats = {} # {'Deepfake': {'total': 0, 'fail': 0}, ...}

        with torch.no_grad():
            for imgs, categories in tqdm(loader, desc=f"Testing {model_name}"):
                imgs = imgs.to(device)
                outputs = model(imgs)
                # Dự đoán: Nếu prob < 0.5 => LIVE (Tức là SAI vì đây là tập Spoof)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).long().cpu().numpy()
                
                for i, pred in enumerate(preds):
                    cat = categories[i]
                    if cat not in stats: stats[cat] = {'total': 0, 'fail': 0}
                    
                    stats[cat]['total'] += 1
                    if pred == 0: # Đoán là Live (0) -> SAI (Miss)
                        stats[cat]['fail'] += 1

        # Lưu vào kết quả chung
        for cat, val in stats.items():
            if cat not in results:
                results[cat] = {'Total_Images': val['total']}
            
            # Tính tỷ lệ lỗi (%)
            error_rate = (val['fail'] / val['total']) * 100 if val['total'] > 0 else 0
            results[cat][model_name] = error_rate

    # --- TỔNG HỢP VÀ VẼ BIEU DO ---
    print("\n--- TONG HOP KET QUA ---")
    df = pd.DataFrame.from_dict(results, orient='index')
    
    # Sắp xếp lại cột cho đẹp
    cols = ['Total_Images'] + [m for m in MODELS.keys() if m in df.columns]
    df = df[cols].fillna(0)
    df = df.sort_values(by='Total_Images', ascending=False)
    
    print(df)
    
    # Lưu CSV
    csv_path = os.path.join(OUTPUT_DIR, "spoof_type_analysis.csv")
    df.to_csv(csv_path)
    print(f"Da luu CSV: {csv_path}")

    # Vẽ Heatmap (Chỉ lấy cột Error Rate của các Model)
    plt.figure(figsize=(10, 6))
    
    # Lấy các cột model (bỏ cột Total_Images)
    heatmap_data = df.drop(columns=['Total_Images'])
    
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd", 
                linewidths=.5, cbar_kws={'label': 'Error Rate (%) - Lower is Better'})
    
    plt.title("Error Rate by Spoof Type (Failure Analysis)", fontsize=14)
    plt.ylabel("Attack Type")
    plt.xlabel("Model")
    plt.tight_layout()
    
    img_path = os.path.join(OUTPUT_DIR, "spoof_type_heatmap.png")
    plt.savefig(img_path)
    print(f"Da luu Heatmap: {img_path}")

if __name__ == "__main__":
    main()