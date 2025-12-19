import os
import csv
import torch
import psutil
import numpy as np
from torch.amp import autocast
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, precision_score, recall_score

def get_system_metrics():
    """Lấy thông tin RAM và VRAM hiện tại"""
    # RAM Usage
    process = psutil.Process(os.getpid())
    ram_usage_mb = process.memory_info().rss / 1024 ** 2
    
    # VRAM Usage (GPU)
    vram_usage_mb = 0
    if torch.cuda.is_available():
        vram_usage_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        
    return ram_usage_mb, vram_usage_mb

def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # TPR @ FPR=1%
    tpr_at_1fpr = 0.0
    if len(np.unique(y_true)) > 1:
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_probs)
        idx = np.abs(fpr_arr - 0.01).argmin()
        tpr_at_1fpr = tpr_arr[idx]
        
    return {
        "acc": acc, "prec": prec, "recall": rec, "f1": f1,
        "tpr_at_1_fpr": tpr_at_1fpr, "cm": cm
    }

def save_log_to_csv(log_dir, epoch, t_met, v_met, time_sec, ram_mb, vram_mb):
    os.makedirs(log_dir, exist_ok=True)
    csv_file = os.path.join(log_dir, "training_log.csv")
    
    # Header cập nhật thêm Memory info
    headers = ["Epoch", "Time(s)", "RAM(MB)", "VRAM(MB)", 
               "Train_Loss", "Train_Acc", 
               "Val_Loss", "Val_Acc", "Val_F1", "Val_TPR@1FPR"]
    
    row = [
        epoch, f"{time_sec:.1f}", f"{ram_mb:.0f}", f"{vram_mb:.0f}",
        f"{t_met['loss']:.4f}", f"{t_met['acc']:.4f}",
        f"{v_met['loss']:.4f}", f"{v_met['acc']:.4f}", f"{v_met['f1']:.4f}", f"{v_met['tpr_at_1_fpr']:.4f}"
    ]
    
    mode = 'a' if os.path.exists(csv_file) else 'w'
    with open(csv_file, mode=mode, newline='') as f:
        writer = csv.writer(f)
        if mode == 'w': writer.writerow(headers)
        writer.writerow(row)

def run_epoch(model, loader, criterion, optimizer, scaler, device, is_train=True, scheduler=None):
    if is_train:
        model.train()
    else:
        model.eval()
        
    running_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device).float()
            
            if is_train:
                optimizer.zero_grad()
                
            with autocast(device_type='cuda', enabled=(scaler is not None)):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                    
            running_loss += loss.item() * imgs.size(0)
            
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
            
    if is_train and scheduler:
        scheduler.step()
        
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    metrics['loss'] = running_loss / len(loader.dataset)
    return metrics