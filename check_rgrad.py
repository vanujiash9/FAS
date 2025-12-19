import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Setup path
import sys
sys.path.append(os.getcwd())
from src.models.build_model.convnext import ConvNextBinary

# ================= CAU HINH =================
MODEL_PATH = "saved_models/convnext/best.pt"
IMG_PATH = "data/data_split/test/1_spoof" # Chon mot anh spoof de test
OUTPUT_DIR = "src/results/paper_figures_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def grad_cam(model, input_tensor, target_layer):
    # Hook de lay feature map va gradient
    feature_maps = []
    gradients = []

    def save_feature_map(module, input, output):
        feature_maps.append(output)
    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_f = target_layer.register_forward_hook(save_feature_map)
    handle_g = target_layer.register_full_backward_hook(save_gradient)

    # Forward
    output = model(input_tensor)
    model.zero_grad()
    
    # Backward cho class spoof
    output.backward()

    # Tinh toan weights
    grads = gradients[0].cpu().data.numpy()
    f_maps = feature_maps[0].cpu().data.numpy()
    
    weights = np.mean(grads, axis=(2, 3))[0]
    cam = np.zeros(f_maps.shape[2:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * f_maps[0, i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    
    handle_f.remove()
    handle_g.remove()
    return cam

def main():
    # 1. Load Model
    model = ConvNextBinary(pretrained=False).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
    model.eval()

    # 2. Lay 1 anh spoof bat ky
    sample_files = [f for f in os.listdir(IMG_PATH) if f.endswith(('.jpg', '.png'))]
    test_file = random.choice(sample_files)
    img_path = os.path.join(IMG_PATH, test_file)
    
    orig_img = Image.open(img_path).convert('RGB')
    img_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(orig_img).unsqueeze(0).to(device)

    # 3. Chon layer cuoi cung cua backbone de soi
    # Voi ConvNext Tiny, layer cuoi thuong la backbone.stages[3]
    target_layer = model.backbone.stages[-1]
    
    cam = grad_cam(model, img_tf, target_layer)

    # 4. Ve va luu
    img_np = np.array(orig_img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Tron anh goc va heatmap
    result = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Anh Goc (Spoof)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title("Grad-CAM (Vung nhan dien)")
    plt.axis('off')

    save_path = os.path.join(OUTPUT_DIR, "Fig9_GradCAM_Explanation.png")
    plt.savefig(save_path, dpi=300)
    print(f"Da luu hinh giai thich tai: {save_path}")

import random
if __name__ == "__main__":
    main()