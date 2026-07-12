"""
Evaluation metrics for MDA prediction.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    top_k = np.argsort(y_score)[::-1][:k]
    return float(y_true[top_k].sum()) / k


def compute_all_metrics(y_true: np.ndarray, y_score: np.ndarray,
                        threshold: float = 0.5,
                        k_list: list = None) -> dict:
    if k_list is None:
        k_list = [10, 20]
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        'auc':  round(roc_auc_score(y_true, y_score), 6),
        'aupr': round(average_precision_score(y_true, y_score), 6),
        'f1':   round(f1_score(y_true, y_pred, zero_division=0), 6),
    }
    for k in k_list:
        metrics[f'precision_at_{k}'] = round(precision_at_k(y_true, y_score, k), 6)
    return metrics
