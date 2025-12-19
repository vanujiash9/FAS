import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve

def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0) # TPR
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion Matrix (TN, FP, FN, TP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Advanced: TPR @ FPR=1%
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_probs)
    idx = np.abs(fpr_arr - 0.01).argmin()
    tpr_at_1fpr = tpr_arr[idx]
    
    return {
        "acc": acc, "prec": prec, "recall": rec, "f1": f1,
        "fpr": fpr, "tpr_at_1_fpr": tpr_at_1fpr,
        "cm": cm
    }