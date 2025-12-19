import os
import time
import yaml
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

from src.models.build_model.vit import ViTBinary
from src.data.data_loader import build_dataloaders
from src.training.utils import run_epoch, save_log_to_csv, get_system_metrics

def main():
    print(">>> DANG KHOI DONG VIT TRAINING...")
    
    config_path = "config/vit.yaml"
    if not os.path.exists(config_path):
        print(f"LOI: Khong tim thay config tai {config_path}")
        return

    with open(config_path) as f: config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir = config['train']['checkpoint_dir']
    results_dir = config['train']['results_dir']
    
    # Dọn dẹp log cũ để tránh lỗi file
    if os.path.exists(results_dir): shutil.rmtree(results_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("Loading Data...")
    loaders = build_dataloaders(**config['dataset'])
    
    model = ViTBinary(pretrained=True).to(device)

    # ViT cần LR thấp (5e-5)
    optimizer = optim.AdamW(model.parameters(), lr=float(config['train']['learning_rate']), weight_decay=float(config['train']['weight_decay']))
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])
    
    best_loss, patience, counter = float('inf'), config['train']['patience'], 0

    print("--- START TRAINING VIT ---")
    try:
        for epoch in range(config['train']['epochs']):
            start = time.time()
            t_met = run_epoch(model, loaders['train'], criterion, optimizer, scaler, device, is_train=True, scheduler=scheduler)
            v_met = run_epoch(model, loaders['test'], criterion, optimizer, scaler, device, is_train=False)
            elapsed = time.time() - start
            
            # Lấy thông số RAM/VRAM
            ram, vram = get_system_metrics()
            
            print(f"Ep {epoch+1}: {elapsed:.0f}s | T_Loss:{t_met['loss']:.4f} Acc:{t_met['acc']:.4f} | V_Loss:{v_met['loss']:.4f} Acc:{v_met['acc']:.4f} | VRAM: {int(vram)}MB")
            
            # Lưu log đầy đủ tham số
            save_log_to_csv(results_dir, epoch+1, t_met, v_met, elapsed, ram, vram)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "last.pt"))

            if v_met['loss'] < best_loss:
                best_loss = v_met['loss']
                counter = 0
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pt"))
                print(">>> SAVED BEST")
            else:
                counter += 1
                if counter >= patience: break
    except KeyboardInterrupt: print("Stopped.")

if __name__ == "__main__":
    main()