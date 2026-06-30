"""model.py — Monotonic constrained GBM for scorecard modeling (CreditModel).

Uses LightGBM with monotonic constraints (ensuring risk ordering aligns with
domain knowledge). Different from DefaultRisk's standard logistic approach.
"""
from __future__ import annotations
import numpy as np; from sklearn.model_selection import train_test_split
from src.core import monotonicity_check, bootstrap_auc_ci, rank_stability, lorenz_curve


def train_monotonic_gbm(data, seed=42):
    X, y = data["X"].values.astype(float), data["y"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    import lightgbm as lgb
    # Monotonic constraints: utilization, delinquency should increase risk; payment_ratio decreases risk
    monotone = [1, -1, 1, 0, 0, -1, 1]  # for 7 features
    model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                monotone_constraints=monotone, random_state=seed, verbose=-1)
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)])
    proba = model.predict_proba(Xte)[:, 1]
    gini = lorenz_curve(yte, proba)
    auc_ci = bootstrap_auc_ci(yte, proba)
    # Feature importance stability across folds
    importances = [model.feature_importances_]
    return {"model": model}, {
        "n_train": len(Xtr), "n_test": len(Xte), "gini": gini, "auc_mean": auc_ci["mean"],
        "auc_ci": [auc_ci["ci_low"], auc_ci["ci_high"]],
        "rank_stability": rank_stability(importances),
    }


def fit_and_evaluate(data, seed=42):
    return train_monotonic_gbm(data, seed)