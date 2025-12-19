import torch
import time
from ptflops import get_model_complexity_info
import sys
import os

sys.path.append(os.getcwd())
from src.models.build_model.convnext import ConvNextBinary
from src.models.build_model.efficientnet import EfficientNetBinary
from src.models.build_model.vit import ViTBinary

def benchmark(name, model_class, size):
    model = model_class(pretrained=False)
    
    # 1. Tinh Parameters va FLOPs
    macs, params = get_model_complexity_info(model, (3, size, size), as_strings=True, print_per_layer_stat=False)
    
    # 2. Tinh thoi gian Inference (FPS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, size, size).to(device)
    
    # Warm up
    for _ in range(10): _ = model(dummy_input)
    
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    end = time.time()
    
    latency = (end - start) / 100 * 1000 # ms
    fps = 1000 / latency
    
    print(f"{name:<15} | {params:<10} | {macs:<10} | {latency:>8.2f} ms | {fps:>6.1f} FPS")

def main():
    print("-" * 65)
    print(f"{'Model':<15} | {'Params':<10} | {'FLOPs':<10} | {'Latency':<10} | {'FPS':<6}")
    print("-" * 65)
    
    benchmark("ConvNeXt", ConvNextBinary, 224)
    benchmark("EfficientNet", EfficientNetBinary, 260)
    benchmark("ViT", ViTBinary, 224)
    print("-" * 65)

if __name__ == "__main__":
    main()
