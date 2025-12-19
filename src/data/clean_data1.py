import os
import csv
import signal
import numpy as np
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from facenet_pytorch import MTCNN

# Paths
ROOT = "data/data_process"
CSV_PATH = f"{ROOT}/dataset_stats/table2_index.csv"
OUTPUT_DIR = f"{ROOT}/cropped_faces"
LOG_DIR = f"{ROOT}/cropped_logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load CSV
def load_index(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    return rows

# GPU setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# MTCNN with optimal settings
mtcnn = MTCNN(
    keep_all=False,
    device=device,
    post_process=False,  # Get raw 0-255 values
    margin=20,  # Padding around face
    thresholds=[0.6, 0.7, 0.7],  # Faster detection
    min_face_size=40  # Skip tiny faces
)

# Graceful shutdown
STOP_SIGNAL = False

def signal_handler(sig, frame):
    global STOP_SIGNAL
    STOP_SIGNAL = True
    print("\nCtrl+C detected, stopping after current batch...")

signal.signal(signal.SIGINT, signal_handler)

# Process single image
def process_image(item):
    if STOP_SIGNAL:
        return "stopped", item["filepath"]
    
    rel_path = item["filepath"]
    img_path = os.path.join(ROOT, rel_path)
    
    try:
        img = Image.open(img_path).convert('RGB')
    except:
        return "cannot_open", rel_path
    
    try:
        # Detect face (returns tensor [C, H, W])
        face_tensor = mtcnn(img)
        
        if face_tensor is None:
            return "no_face", rel_path
        
        # Convert: [C,H,W] -> [H,W,C]
        face_array = face_tensor.permute(1, 2, 0).cpu().numpy()
        face_array = np.clip(face_array, 0, 255).astype(np.uint8)
        
        # Save
        face_img = Image.fromarray(face_array)
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(rel_path))
        face_img.save(out_path, quality=95)
        
        return "ok", rel_path
        
    except Exception as e:
        return "error", rel_path

# Main
def main():
    global STOP_SIGNAL
    
    print("="*60)
    print("GPU-ACCELERATED FACE CROPPING")
    print("="*60)
    
    index = load_index(CSV_PATH)
    print(f"\nTotal images: {len(index):,}")
    
    # Optimize workers based on device
    if device == 'cuda':
        max_workers = 4  # GPU works better with fewer threads
    else:
        max_workers = 8  # CPU can handle more threads
    
    print(f"Workers: {max_workers}\n")
    
    results = {"cannot_open": [], "no_face": [], "error": [], "stopped": []}
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_image, item) for item in index]
            
            for f in tqdm(as_completed(futures), total=len(futures), 
                         desc="Processing", unit="img"):
                if STOP_SIGNAL:
                    print("\nCancelling remaining tasks...")
                    break
                
                status, rel_path = f.result()
                if status != "ok":
                    results[status].append(rel_path)
    
    except KeyboardInterrupt:
        print("\nForced shutdown...")
    
    # Save logs
    for status, items in results.items():
        if items:
            log_path = os.path.join(LOG_DIR, f"{status}.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                for filepath in items:
                    f.write(filepath + "\n")
    
    # Calculate stats correctly
    total_failed = sum(len(v) for v in results.values())
    successful = len(index) - total_failed
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total: {len(index):,}")
    print(f"Success: {successful:,} ({successful/len(index)*100:.1f}%)")
    print(f"Failed: {total_failed:,}")
    
    if results["cannot_open"]:
        print(f"  - Cannot open: {len(results['cannot_open']):,}")
    if results["no_face"]:
        print(f"  - No face: {len(results['no_face']):,}")
    if results["error"]:
        print(f"  - Error: {len(results['error']):,}")
    if results["stopped"]:
        print(f"  - Stopped: {len(results['stopped']):,}")
    
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Logs: {LOG_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()