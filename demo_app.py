import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from facenet_pytorch import MTCNN
import requests
from io import BytesIO

# =========================
# 1. Cấu hình
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=device)

MODELS = {
    "ConvNeXt": {"class": "ConvNextBinary", "path": "saved_models/convnext/best.pt", "size": 224},
    "EfficientNet": {"class": "EfficientNetBinary", "path": "checkpoints/efficientnet/best.pt", "size": 260},
    "ViT": {"class": "ViTBinary", "path": "checkpoints/vit/best.pt", "size": 224}
}

RESULT_DIR = "PAD/src/results/demo"
os.makedirs(RESULT_DIR, exist_ok=True)

# =========================
# 2. Load model
# =========================
def load_model(cls, path):
    from importlib import import_module
    # Map class name to correct module file
    class_to_module = {
        "ConvNextBinary": "convnext",
        "EfficientNetBinary": "efficientnet",
        "ViTBinary": "vit"
    }
    module_name = class_to_module.get(cls, cls.lower())
    module = import_module(f"src.models.build_model.{module_name}")
    model_class = getattr(module, cls)
    model = model_class(pretrained=False).to(device)
    state = torch.load(path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model

# =========================
# 3. Load ảnh từ file hoặc URL
# =========================
def load_image(path_or_url):
    path_or_url = path_or_url.strip()
    if path_or_url.startswith("http"):
        try:
            resp = requests.get(path_or_url, timeout=10)
            image = Image.open(BytesIO(resp.content)).convert("RGB")
            return image
        except Exception as e:
            print(f"Lỗi load ảnh từ URL: {e}")
            return None
    else:
        if os.path.exists(path_or_url):
            return Image.open(path_or_url).convert("RGB")
        else:
            print(f"File không tồn tại: {path_or_url}")
            return None

# =========================
# 4. Xử lý ảnh
# =========================
def process_image(image, model, size, threshold):
    img_np = np.array(image)
    boxes, _ = mtcnn.detect(image)
    draw = ImageDraw.Draw(image)
    labels = []

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    if boxes is None:
        return image, labels

    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        margin = int(0.2 * (x2 - x1))
        x1, y1 = max(0, x1-margin), max(0, y1-margin)
        x2, y2 = min(image.width, x2+margin), min(image.height, y2+margin)
        face = image.crop((x1, y1, x2, y2))
        inp = transform(face).unsqueeze(0).to(device)
        with torch.no_grad():
            score = torch.sigmoid(model(inp)).item()
        is_spoof = score > threshold
        label = "SPOOF" if is_spoof else "REAL"
        color = "red" if is_spoof else "green"
        conf = score if is_spoof else 1-score
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1, y1-20), f"{label} {conf:.2%}", fill=color)
        labels.append(label)

    return image, labels

# =========================
# 5. Main
# =========================
if __name__ == "__main__":
    paths = input("Nhập đường dẫn file hoặc URL (nhiều link cách nhau bằng ','): ").split(",")
    threshold_input = input("Nhập threshold (0-1, mặc định 0.5): ")
    threshold = float(threshold_input) if threshold_input else 0.5

    for p in paths:
        p = p.strip()
        if not p:
            continue
        print(f"\n=== Xử lý ảnh: {p} ===")
        image = load_image(p)
        if image is None:
            continue
        fname = os.path.basename(p.split("?")[0])

        for model_name, cfg in MODELS.items():
            print(f"\n--- Kiểm tra model {model_name} ---")
            try:
                model = load_model(cfg["class"], cfg["path"])
                out_img, labels = process_image(image.copy(), model, cfg["size"], threshold)
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
