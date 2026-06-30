"""data.py — Credit card default data for scorecard modeling (Aptivaa).

Different from DefaultRisk (mortgage PD): uses revolving credit features
with utilization rates, payment history, and revolving balances.
"""
from __future__ import annotations
import numpy as np; import pandas as pd


def make_synthetic(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "utilization_rate": rng.beta(2, 5, n).round(3),
        "age_months": rng.integers(6, 360, n),
        "delinquency_days": rng.weibull(0.8, n).clip(0, 180).round(0),
        "num_trades": rng.poisson(8, n).clip(1, 40),
        "revolving_balance": rng.lognormal(9, 1, n).clip(0, 100000).round(0),
        "payment_ratio": rng.beta(5, 2, n).round(3),
        "inquiries_6m": rng.poisson(0.5, n).clip(0, 10),
    })
    util = df["utilization_rate"]; age = np.clip(df["age_months"] / 360, 0, 1)
    delinq = np.clip(df["delinquency_days"] / 180, 0, 1); trades = np.clip(df["num_trades"] / 40, 0, 1)
    pay = df["payment_ratio"]; bal = np.clip(df["revolving_balance"] / 100000, 0, 1)
    inq = np.clip(df["inquiries_6m"] / 10, 0, 1)
    log_odds = -1 + 3*util + 1.5*delinq - 0.8*age + 0.3*trades - 1.5*pay + 0.4*bal + 0.6*inq + rng.normal(0, 0.5, n)
    prob = 1/(1+np.exp(-log_odds)); y = (prob > np.percentile(prob, 70)).astype(float)
    return {"X": df, "y": y, "n_samples": n, "n_features": len(df.columns), "positive_rate": y.mean(),
            "categorical_features": [], "numerical_features": list(df.columns)}