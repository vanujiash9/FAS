import os
import sys
import torch

# Dinh nghia duong dan den cac file code model
paths = {
    "convnext": "src/models/build_model/convnext.py",
    "efficientnet": "src/models/build_model/efficientnet.py",
    "vit": "src/models/build_model/vit.py"
}

def check_files():
    print("--- KIEM TRA FILE CODE ---")
    all_exist = True
    for name, path in paths.items():
        if os.path.exists(path):
            print(f"[OK] Tim thay file: {path}")
        else:
            print(f"[MISSING] Khong tim thay file: {path}")
            all_exist = False
    return all_exist

def check_models():
    print("\n--- KIEM TRA TAI MODEL (PRETRAINED) ---")
    
    # Them thu muc hien tai vao path de import duoc src
    sys.path.append(os.getcwd())

    try:
        print("1. Dang tai ConvNeXt...", end=" ", flush=True)
        from src.models.build_model.convnext import ConvNextBinary
        model = ConvNextBinary(pretrained=True)
        print("[OK]")
    except Exception as e:
        print(f"\n[ERROR] ConvNeXt: {e}")

    try:
        print("2. Dang tai EfficientNet...", end=" ", flush=True)
        from src.models.build_model.efficientnet import EfficientNetBinary
        model = EfficientNetBinary(pretrained=True)
        print("[OK]")
    except Exception as e:
        print(f"\n[ERROR] EfficientNet: {e}")

    try:
        print("3. Dang tai ViT...", end=" ", flush=True)
        from src.models.build_model.vit import ViTBinary
        model = ViTBinary(pretrained=True)
        print("[OK]")
    except Exception as e:
        print(f"\n[ERROR] ViT: {e}")

if __name__ == "__main__":
    if check_files():
        check_models()
    else:
        print("\nVui long tao cac file model bi thieu truoc khi tiep tuc.")