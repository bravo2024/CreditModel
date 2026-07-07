"""data.py — Credit card default data for scorecard modeling.

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
    return {"X": df, "y": pd.Series(y), "n_samples": n, "n_features": len(df.columns), "positive_rate": y.mean(),
            "categorical_features": [], "numerical_features": list(df.columns)}

def load_taiwan_default(cache_dir=None):
    """UCI 'default of credit card clients' (Yeh & Lien 2009): 30k Taiwanese
    revolving-credit accounts, one month of default outcomes.

    Fetched from OpenML (data id 42477) and cached as a local csv. Features are
    renamed per the UCI codebook and two scorecard ratios are derived:
    utilization (last bill / limit) and payment ratio (last payment / last bill).
    """
    from pathlib import Path

    cache = Path(cache_dir or Path(__file__).parent.parent / "data") / "taiwan_default.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        from sklearn.datasets import fetch_openml
        raw = fetch_openml(data_id=42477, as_frame=True, parser="auto").frame
        names = (["limit_bal", "sex", "education", "marriage", "age",
                  "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
                 + [f"bill_amt{i}" for i in range(1, 7)]
                 + [f"pay_amt{i}" for i in range(1, 7)] + ["default"])
        raw.columns = names
        df = raw.astype(float)
        cache.parent.mkdir(exist_ok=True)
        df.to_csv(cache, index=False)

    df["utilization_rate"] = (df["bill_amt1"] / df["limit_bal"]).clip(0, 2).round(3)
    df["payment_ratio"] = (df["pay_amt1"] / df["bill_amt1"].clip(lower=1)).clip(0, 2).round(3)

    features = ["limit_bal", "age", "pay_0", "pay_2", "pay_3",
                "bill_amt1", "pay_amt1", "utilization_rate", "payment_ratio"]
    X = df[features].copy()
    y = df["default"].to_numpy(dtype=float)
    return {"X": X, "y": pd.Series(y), "n_samples": len(df), "n_features": len(features),
            "positive_rate": float(y.mean()), "categorical_features": [],
            "numerical_features": features}
