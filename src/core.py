"""core.py — Scorecard validation metrics for CreditModel.

NOT the generic metrics. Uses:
  * Monotonicity check — ensures risk ordering constraints
  * Bootstrap AUC with CI — confidence intervals for discrimination
  * Variable Importance stability — rank agreement across folds
  * Reject inference metrics — augmented model comparison

References: Siddiqi (2006) "Credit Risk Scorecards"; Basel II guidelines.
"""
from __future__ import annotations
import numpy as np; from scipy.stats import spearmanr


def monotonicity_check(coefficients, expected_signs):
    """Check if model coefficients have the expected monotonic direction."""
    if len(coefficients) != len(expected_signs):
        return 0.0
    ok = sum(1 for c, s in zip(coefficients, expected_signs) if c * s >= 0)
    return ok / max(len(coefficients), 1)


def bootstrap_auc_ci(y_true, y_proba, n_iter=100):
    """Bootstrap confidence interval for AUC."""
    aucs = []
    idx = np.arange(len(y_true))
    for _ in range(n_iter):
        s = np.random.choice(idx, len(idx), replace=True)
        y, p = y_true[s], y_proba[s]
        if len(np.unique(y)) < 2:
            continue
        from sklearn.metrics import roc_auc_score as ras
        aucs.append(ras(y, p))
    return {"mean": float(np.mean(aucs)), "ci_low": float(np.percentile(aucs, 2.5)), "ci_high": float(np.percentile(aucs, 97.5))}


def rank_stability(fold_importances):
    """Top-k vs other rank agreement using Spearman correlation."""
    n = len(fold_importances)
    if n < 2:
        return 1.0
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            c, _ = spearmanr(fold_importances[i], fold_importances[j])
            if np.isfinite(c):
                corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0


def lorenz_curve(y_true, y_proba):
    """Gini = 2*AUC - 1 for credit risk discrimination."""
    from sklearn.metrics import roc_auc_score
    return float(2 * roc_auc_score(y_true, y_proba) - 1)

def population_stability_index(expected, actual, n_bins=10):
    """PSI between a development-time score distribution and a current one."""
    bins = np.linspace(0, 1, n_bins + 1)
    e, _ = np.histogram(expected, bins=bins)
    a, _ = np.histogram(actual, bins=bins)
    e = np.clip(e / max(e.sum(), 1), 1e-6, None)
    a = np.clip(a / max(a.sum(), 1), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
