"""
evaluation.py — Cross-validation metrics and reporting utilities.
"""

import numpy as np
from typing import List, Dict


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    Compute AUC and AUPR from true labels and predicted scores.

    Parameters
    ----------
    y_true  : array-like of 0/1 labels
    y_score : array-like of predicted probabilities

    Returns
    -------
    dict with keys: auc, aupr
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    return {
        "auc":  roc_auc_score(y_true, y_score),
        "aupr": average_precision_score(y_true, y_score),
    }


def print_cv_summary(fold_results: List[Dict]) -> None:
    """
    Print a formatted cross-validation summary table.

    Parameters
    ----------
    fold_results : list of dicts with keys fold, auc, aupr, n_train, n_test
    """
    print("\n" + "=" * 65)
    print(f"{'Fold':>5}  {'AUC':>8}  {'AUPR':>8}  {'N_train':>8}  {'N_test':>8}")
    print("-" * 65)
    aucs, auprs = [], []
    for r in fold_results:
        print(f"{r['fold']:>5}  {r['auc']:.4f}    {r['aupr']:.4f}    "
              f"{r['n_train']:>8}  {r['n_test']:>8}")
        aucs.append(r['auc'])
        auprs.append(r['aupr'])
    print("-" * 65)
    print(f"{'Mean':>5}  {np.mean(aucs):.4f}    {np.mean(auprs):.4f}")
    print(f"{'  SD':>5}  {np.std(aucs, ddof=1):.4f}    {np.std(auprs, ddof=1):.4f}")
    print("=" * 65)

    # Wilcoxon test vs baseline if available
    baselines = [r.get("baseline_auc", None) for r in fold_results]
    if any(b is not None and b > 0 for b in baselines):
        from scipy.stats import wilcoxon
        try:
            stat, p = wilcoxon(aucs, baselines)
            print(f"\nWilcoxon signed-rank test vs LR baseline: "
                  f"p = {p:.4f}  (n = {len(aucs)} folds)")
        except Exception:
            pass


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "auc",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap 95% confidence interval for AUC or AUPR.

    Parameters
    ----------
    y_true, y_score : labels and scores
    metric : 'auc' or 'aupr'
    n_boot : number of bootstrap resamples
    alpha  : significance level (0.05 for 95% CI)
    seed   : random seed

    Returns
    -------
    dict: point_estimate, ci_lower, ci_upper
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    rng = np.random.default_rng(seed)
    fn  = roc_auc_score if metric == "auc" else average_precision_score
    n   = len(y_true)

    point = fn(y_true, y_score)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_vals.append(fn(y_true[idx], y_score[idx]))

    lo = np.percentile(boot_vals, 100 * alpha / 2)
    hi = np.percentile(boot_vals, 100 * (1 - alpha / 2))
    return {"point_estimate": point, "ci_lower": lo, "ci_upper": hi}
