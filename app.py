import streamlit as st
import torch
import cv2
import numpy as np
import os
import sys
from PIL import Image, ImageDraw
from torchvision import transforms
import traceback

# --- 1. SETUP PATH ---
BASE_DIR = os.getcwd()
sys.path.append(BASE_DIR)

# Import model architectures
from src.models.build_model.convnext import ConvNextBinary
from src.models.build_model.efficientnet import EfficientNetBinary
from src.models.build_model.vit import ViTBinary

# --- 2. CONFIG STREAMLIT ---
st.set_page_config(page_title="Face Anti-Spoof Demo", layout="wide")
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 3. MODEL CONFIG ---
MODELS_CONFIG = {
    "ConvNeXt": {
        "class": ConvNextBinary,
        "path": os.path.join(BASE_DIR, "saved_models/convnext/best.pt"),
        "size": 224
    },
    "EfficientNet": {
        "class": EfficientNetBinary,
        "path": os.path.join(BASE_DIR, "checkpoints/efficientnet/best.pt"),
        "size": 260
    },
    "ViT": {
        "class": ViTBinary,
        "path": os.path.join(BASE_DIR, "checkpoints/vit/best.pt"),
        "size": 224
    }
}

# --- 4. LOAD MODEL & FACE DETECTOR ---
# Không dùng cache để debug dễ dàng
def load_resources(model_name):
    cfg = MODELS_CONFIG[model_name]
    weight_path = cfg['path']
    
    st.text(f"Đang load model từ: {weight_path}")
    
    if not os.path.exists(weight_path):
        return None, None, f"File weights khong ton tai: {weight_path}"
    
    try:
        # 1. Load model
        model = cfg['class'](pretrained=False).to(device)
        
        # 2. Load weights
        state_dict = torch.load(weight_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        
        # 3. Load Haar cascade face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        st.text("Model load thanh cong!")
        return model, face_cascade, cfg['size']
    
    except Exception as e:
        return None, None, traceback.format_exc()

# --- 5. PROCESS IMAGE ---
def process_image(image_pil, model, face_cascade, input_size, threshold):
    img_cv = np.array(image_pil)
    if img_cv.shape[-1] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
    draw = ImageDraw.Draw(image_pil)
    results_info = []
    
    tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
    ])
    
    for (x, y, w, h) in faces:
        m = int(w * 0.2)
        h_img, w_img, _ = img_cv.shape
        x1, y1 = max(0, x-m), max(0, y-m)
        x2, y2 = min(w_img, x+w+m), min(h_img, y+h+m)
        face_crop = image_pil.crop((x1, y1, x2, y2))
        
        inp = tf(face_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            score = torch.sigmoid(model(inp)).item()
        
        is_spoof = score > threshold
        label = "SPOOF" if is_spoof else "REAL"
        color = "red" if is_spoof else "green"
        conf = score if is_spoof else 1 - score
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1, y1-15), f"{label} {conf:.1%}", fill=color)
        
        results_info.append(label)
    
    return image_pil, results_info

# --- 6. STREAMLIT APP ---
def main():
    st.title("🛡️ Face Anti-Spoofing Demo")
    
    with st.sidebar:
        st.header("Cau Hinh")
        model_choice = st.selectbox("Chon Model", list(MODELS_CONFIG.keys()))
        threshold = st.slider("Nguong (Threshold)", 0.0, 1.0, 0.5, 0.01)
        st.caption(f"Device: {device}")
    
    # Load model
    model, detector, size_or_err = load_resources(model_choice)
    if model is None:
        st.error(f"Loi khi load model:\n{size_or_err}")
        st.stop()
    
    uploaded_file = st.file_uploader("Upload anh (JPG/PNG)", type=['jpg','jpeg','png'])
    if uploaded_file:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file).convert('RGB')
        col1.image(image, caption="Anh goc", use_container_width=True)
        
        if st.button("Kiem Tra Ngay"):
            with st.spinner("Dang xu ly..."):
                res_img, info = process_image(image.copy(), model, detector, size_or_err, threshold)
            
            col2.image(res_img, caption="Ket qua", use_container_width=True)
            
            if not info:
                st.warning("Khong tim thay khuon mat!")
            elif "SPOOF" in info:
                st.error(f"Phat hien {info.count('SPOOF')} mat gia mao!")
            else:
                st.success("Tat ca khuon mat deu la THAT (REAL).")

if __name__ == "__main__":
    main()
PAD/saved_models/convnext/last.pt