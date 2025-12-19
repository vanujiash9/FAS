import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import requests
from io import BytesIO

sys.path.append(os.getcwd())

from src.models.build_model.convnext import ConvNextBinary
from src.models.build_model.efficientnet import EfficientNetBinary
from src.models.build_model.vit import ViTBinary

device = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "ConvNeXt": {"class": ConvNextBinary, "path": "saved_models/convnext/best.pt", "size": 224},
    "EfficientNet": {"class": EfficientNetBinary, "path": "checkpoints/efficientnet/best.pt", "size": 260},
    "ViT": {"class": ViTBinary, "path": "checkpoints/vit/best.pt", "size": 224}
}

# Thư mục lưu kết quả
RESULT_DIR = "PAD/src/results/demo"
os.makedirs(RESULT_DIR, exist_ok=True)

# Load model
def load_model(cfg):
    model = cfg["class"](pretrained=False).to(device)
    state = torch.load(cfg["path"], map_location=device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return model, detector

# Xử lý ảnh
def process_image(image, model, detector, size, threshold):
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 4, minSize=(50,50))
    draw = ImageDraw.Draw(image)
    labels = []

    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    for (x,y,w,h) in faces:
        m = int(0.2*w)
        x1, y1 = max(0, x-m), max(0, y-m)
        x2, y2 = min(img_np.shape[1], x+w+m), min(img_np.shape[0], y+h+m)
        face = image.crop((x1,y1,x2,y2))
        inp = transform(face).unsqueeze(0).to(device)
        with torch.no_grad():
            score = torch.sigmoid(model(inp)).item()
        is_spoof = score > threshold
        label = "SPOOF" if is_spoof else "REAL"
        color = "red" if is_spoof else "green"
        conf = score if is_spoof else 1-score
        draw.rectangle([x1,y1,x2,y2], outline=color, width=4)
        draw.text((x1, y1-20), f"{label} {conf:.2%}", fill=color)
        labels.append(label)
    return image, labels

def load_image(path_or_url):
    if path_or_url.startswith("http"):
        try:
            resp = requests.get(path_or_url)
            image = Image.open(BytesIO(resp.content)).convert("RGB")
            return image
        except Exception as e:
            print(f"Lỗi load ảnh: {e}")
            return None
    else:
        if os.path.exists(path_or_url):
            return Image.open(path_or_url).convert("RGB")
        else:
            print(f"File không tồn tại: {path_or_url}")
            return None

if __name__ == "__main__":
    paths = input("Nhập đường dẫn file hoặc URL (nhiều link cách nhau bằng ','): ").split(",")
    threshold = input("Nhập threshold (0-1, mặc định 0.5): ")
    threshold = float(threshold) if threshold else 0.5

    for p in paths:
        p = p.strip()
        print(f"\n=== Xử lý ảnh: {p} ===")
        image = load_image(p)
        if image is None:
            continue
        fname = os.path.basename(p.split("?")[0])  # lấy tên file
        for model_name, cfg in MODELS.items():
            print(f"\n--- Kiểm tra model {model_name} ---")
            try:
                model, detector = load_model(cfg)
                out_img, labels = process_image(image.copy(), model, detector, cfg['size'], threshold)
                # Lưu kết quả
                out_path = os.path.join(RESULT_DIR, f"{os.path.splitext(fname)[0]}_{model_name}.jpg")
                out_img.save(out_path)
                if not labels:
                    print("⚠️ Không tìm thấy khuôn mặt")
                elif "SPOOF" in labels:
                    print(f"🚨 Phát hiện {labels.count('SPOOF')} khuôn mặt giả")
                else:
                    print(f"✅ {len(labels)} khuôn mặt đều REAL")
                print(f"Lưu kết quả tại: {out_path}")
            except Exception as e:
                print(f"Lỗi khi xử lý model {model_name}: {e}")
