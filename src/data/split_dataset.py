import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ================= CẤU HÌNH =================
SPLIT_DIR = os.path.join('data', 'data_split')
OUTPUT_DIR = os.path.join('data', 'data_process')

# Định nghĩa nhóm tấn công để thống kê cho gọn
SPOOF_CATEGORIES = {
    'Deepfake': ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter', 'CelebA'],
    'Mask': ['silicon', 'mask', 'latex', 'UpperBodyMask', 'RegionMask', '3D_Mask'],
    'Print': ['Poster', 'Photo', 'A4', 'Print'],
    'Replay': ['Phone', 'PC', 'Pad', 'Screen', 'Replay']
}

def get_spoof_type_from_filename(filename):
    """
    Parse tên file: Dataset_ID_Label_Type.ext
    """
    try:
        name_body = os.path.splitext(filename)[0]
        parts = name_body.split('_')
        # Loại spoof nằm ở cuối cùng
        raw_type = parts[-1]
        
        # Gom nhóm (Mapping)
        for category, keywords in SPOOF_CATEGORIES.items():
            for kw in keywords:
                if kw.lower() in raw_type.lower():
                    return category
        
        # Nếu không thuộc nhóm nào thì trả về tên gốc
        return raw_type
    except:
        return "Unknown"

def collect_stats(subset_name):
    """Quét thư mục train hoặc test để đếm"""
    subset_path = os.path.join(SPLIT_DIR, subset_name)
    stats = {
        'Live': 0,
        'Spoof': 0,
        'Spoof_Types': defaultdict(int)
    }
    
    if not os.path.exists(subset_path):
        return stats

    # 1. Đếm Live
    live_path = os.path.join(subset_path, '0_live')
    if os.path.exists(live_path):
        files = os.listdir(live_path)
        stats['Live'] = len(files)

    # 2. Đếm Spoof và phân loại
    spoof_path = os.path.join(subset_path, '1_spoof')
    if os.path.exists(spoof_path):
        files = os.listdir(spoof_path)
        stats['Spoof'] = len(files)
        
        for f in files:
            s_type = get_spoof_type_from_filename(f)
            stats['Spoof_Types'][s_type] += 1
            
    return stats

def main():
    if not os.path.exists(SPLIT_DIR):
        print(f"Loi: Khong tim thay {SPLIT_DIR}")
        return

    print("Dang thong ke du lieu Train/Test...")
    
    train_stats = collect_stats('train')
    test_stats = collect_stats('test')

    # ================= BÁO CÁO TEXT =================
    print("\n" + "="*60)
    print("BAO CAO CHI TIET PHAN BO DU LIEU (TRAIN vs TEST)")
    print("="*60)
    
    # 1. Tổng quan
    total_train = train_stats['Live'] + train_stats['Spoof']
    total_test = test_stats['Live'] + test_stats['Spoof']
    
    print(f"{'SET':<10} | {'LIVE':<10} | {'SPOOF':<10} | {'TOTAL':<10} | {'RATIO (L:S)'}")
    print("-" * 65)
    
    r_train = f"{train_stats['Live']/train_stats['Spoof']:.1f} : 1" if train_stats['Spoof'] else "N/A"
    print(f"{'TRAIN':<10} | {train_stats['Live']:<10} | {train_stats['Spoof']:<10} | {total_train:<10} | {r_train}")
    
    r_test = f"{test_stats['Live']/test_stats['Spoof']:.1f} : 1" if test_stats['Spoof'] else "N/A"
    print(f"{'TEST':<10} | {test_stats['Live']:<10} | {test_stats['Spoof']:<10} | {total_test:<10} | {r_test}")
    
    print("-" * 65)
    print("\nCHI TIET CAC LOAI TAN CONG (SPOOF TYPES):")
    print(f"{'TYPE':<20} | {'TRAIN QTY':<10} | {'TEST QTY':<10} | {'BALANCE?'}")
    print("-" * 65)
    
    # Lấy tất cả các loại spoof có trong cả 2 tập
    all_types = set(list(train_stats['Spoof_Types'].keys()) + list(test_stats['Spoof_Types'].keys()))
    sorted_types = sorted(list(all_types))
    
    for t in sorted_types:
        c_train = train_stats['Spoof_Types'][t]
        c_test = test_stats['Spoof_Types'][t]
        
        # Đánh giá độ cân bằng (Train thường gấp ~4 lần Test là đẹp vì tỷ lệ 80/20)
        # Nếu test quá ít (<5) thì bỏ qua đánh giá
        status = "OK"
        if c_train > 0 and c_test == 0: status = "MISSING IN TEST"
        elif c_train == 0 and c_test > 0: status = "MISSING IN TRAIN"
        
        print(f"{t:<20} | {c_train:<10} | {c_test:<10} | {status}")

    # ================= VẼ BIỂU ĐỒ =================
    print("\nDang ve bieu do so sanh...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Biểu đồ 1: Số lượng Live vs Spoof
    labels = ['Live', 'Spoof']
    train_vals = [train_stats['Live'], train_stats['Spoof']]
    test_vals = [test_stats['Live'], test_stats['Spoof']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x - width/2, train_vals, width, label='Train', color='#4CAF50')
    ax1.bar(x + width/2, test_vals, width, label='Test', color='#FF9800')
    
    ax1.set_ylabel('So luong anh')
    ax1.set_title('So sanh so luong Live/Spoof')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    
    # Biểu đồ 2: Phân bố loại Spoof (Normalized %)
    # Để xem tỷ lệ thành phần các loại tấn công có giống nhau không
    
    # Chỉ lấy top 5 loại phổ biến nhất để vẽ cho thoáng
    top_types = sorted(all_types, key=lambda t: train_stats['Spoof_Types'][t], reverse=True)[:5]
    
    train_counts = [train_stats['Spoof_Types'][t] for t in top_types]
    test_counts = [test_stats['Spoof_Types'][t] for t in top_types]
    
    # Chuyển sang phần trăm (%) để dễ so sánh phân phối
    total_spoof_train = sum(train_counts) if sum(train_counts) > 0 else 1
    total_spoof_test = sum(test_counts) if sum(test_counts) > 0 else 1
    
    train_pct = [x / total_spoof_train * 100 for x in train_counts]
    test_pct = [x / total_spoof_test * 100 for x in test_counts]
    
    x2 = np.arange(len(top_types))
    
    ax2.bar(x2 - width/2, train_pct, width, label='Train (%)', color='#2196F3')
    ax2.bar(x2 + width/2, test_pct, width, label='Test (%)', color='#E91E63')
    
    ax2.set_ylabel('Ty le (%) trong tap Spoof')
    ax2.set_title('Phan bo cac loai tan cong (Top 5)')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(top_types)
    ax2.legend()
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'split_statistics.png')
    plt.savefig(save_path)
    print(f"\nDa luu bieu do tai: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()