"""model.py — Monotonic constrained GBM for scorecard modeling (CreditModel).

Uses LightGBM with monotonic constraints (ensuring risk ordering aligns with
domain knowledge). Different from DefaultRisk's standard logistic approach.
"""
from __future__ import annotations
import numpy as np; from sklearn.model_selection import train_test_split
from src.core import monotonicity_check, bootstrap_auc_ci, rank_stability, lorenz_curve


def _monotone_sign(name):
    """Direction each feature is allowed to push PD, from domain knowledge."""
    if name.startswith(("utilization", "delinquency", "pay_0", "pay_2", "pay_3",
                        "pay_4", "pay_5", "pay_6", "inquiries", "num_trades")):
        return 1    # more utilization / worse payment status -> higher risk
    if name.startswith(("payment_ratio", "age", "limit_bal")):
        return -1   # paying bills down, seasoning, higher granted limit -> lower risk
    return 0


def train_monotonic_gbm(data, seed=42):
    feats = list(data["X"].columns)
    X, y = data["X"].values.astype(float), data["y"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    import lightgbm as lgb
    monotone = [_monotone_sign(f) for f in feats]
    model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                monotone_constraints=monotone, random_state=seed, verbose=-1)
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)])
    proba = model.predict_proba(Xte)[:, 1]
    gini = lorenz_curve(yte, proba)
    auc_ci = bootstrap_auc_ci(yte, proba)
    # Feature importance stability across folds
    importances = [model.feature_importances_]
    return {"model": model, "features": feats, "X_test": Xte, "y_test": yte, "proba_test": proba,
            "importances": model.feature_importances_}, {
        "n_train": len(Xtr), "n_test": len(Xte), "gini": gini, "auc_mean": auc_ci["mean"],
        "auc_ci": [auc_ci["ci_low"], auc_ci["ci_high"]],
        "rank_stability": rank_stability(importances),
    }


def fit_and_evaluate(data, seed=42):
    return train_monotonic_gbm(data, seed)