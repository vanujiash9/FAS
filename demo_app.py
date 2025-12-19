import streamlit as st
import torch
import cv2
import numpy as np
import os
import sys
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# =========================
# 1. SETUP
# =========================
sys.path.append(os.getcwd())

from src.models.build_model.convnext import ConvNextBinary
from src.models.build_model.efficientnet import EfficientNetBinary
from src.models.build_model.vit import ViTBinary

st.set_page_config(
    page_title="Face Anti-Spoofing Demo",
    layout="wide"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "ConvNeXt": {
        "class": ConvNextBinary,
        "path": "saved_models/convnext/best.pt",
        "size": 224
    },
    "EfficientNet": {
        "class": EfficientNetBinary,
        "path": "checkpoints/efficientnet/best.pt",
        "size": 260
    },
    "ViT": {
        "class": ViTBinary,
        "path": "checkpoints/vit/best.pt",
        "size": 224
    }
}

# =========================
# 2. LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_resources(model_name):
    cfg = MODELS[model_name]

    if not os.path.exists(cfg["path"]):
        return None, None, None, f"Không tìm thấy {cfg['path']}"

    model = cfg["class"](pretrained=False)
    model.to(device)

    state = torch.load(cfg["path"], map_location=device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    return model, face_detector, cfg["size"], None

# =========================
# 3. MAIN APP
# =========================
def main():
    st.title("🛡️ Face Anti-Spoofing Demo")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        model_name = st.selectbox("Chọn mô hình", MODELS.keys())
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)
        st.caption(f"Device: `{device}`")

    model, detector, img_size, err = load_resources(model_name)

    if err:
        st.error(err)
        st.stop()

    uploaded_file = st.file_uploader(
        "📤 Upload ảnh khuôn mặt",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("⬆️ Hãy upload ảnh để bắt đầu")
        return

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")
    col1.image(image, caption="Ảnh gốc", use_container_width=True)

    if st.button("🔍 Phân tích", type="primary"):
        with st.spinner("Đang xử lý..."):
            img_np = np.array(image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(50, 50)
            )

            if len(faces) == 0:
                st.warning("⚠️ Không phát hiện khuôn mặt")
                return

            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            draw = ImageDraw.Draw(image)
            results = []

            for (x, y, w, h) in faces:
                margin = int(0.2 * w)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(img_np.shape[1], x + w + margin)
                y2 = min(img_np.shape[0], y + h + margin)

                face = image.crop((x1, y1, x2, y2))
                inp = transform(face).unsqueeze(0).to(device)

                with torch.no_grad():
                    score = torch.sigmoid(model(inp)).item()

                is_spoof = score > threshold
                label = "SPOOF" if is_spoof else "REAL"
                conf = score if is_spoof else 1 - score
                color = "red" if is_spoof else "green"

                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                draw.text(
                    (x1, y1 - 20),
                    f"{label} {conf:.2%}",
                    fill=color
                )

                results.append(label)

            col2.image(image, caption="Kết quả", use_container_width=True)

            if "SPOOF" in results:
                st.error(f"🚨 Phát hiện {results.count('SPOOF')} khuôn mặt giả")
            else:
                st.success(f"✅ {len(results)} khuôn mặt đều THẬT")

# =========================
if __name__ == "__main__":
    main()
