from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["pd_score","exposure_at_default","loss_given_default","credit_rating","industry_sector","loan_tenure_months","collateral_ratio","country_risk","maturity_years","restructured_flag"]
CATEGORICAL_FEATURES = ["credit_rating","industry_sector","restructured_flag"]
NUMERICAL_FEATURES = ["pd_score","exposure_at_default","loss_given_default","loan_tenure_months","collateral_ratio","country_risk","maturity_years"]
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "pd_score": rng.beta(2,8,size=n).round(4),
        "exposure_at_default": rng.lognormal(mean=12,sigma=0.8,size=n).clip(10000,10_000_000).astype(int),
        "loss_given_default": rng.beta(4,5,size=n).round(3),
        "credit_rating": rng.choice(["AAA","AA","A","BBB","BB","B","CCC","D"],size=n,p=[0.05,0.10,0.15,0.25,0.20,0.12,0.08,0.05]),
        "industry_sector": rng.choice(["manufacturing","services","technology","energy","healthcare","finance"],size=n,p=[0.20,0.18,0.15,0.12,0.18,0.17]),
        "loan_tenure_months": rng.uniform(12,360,size=n).astype(int),
        "collateral_ratio": rng.uniform(0,2,size=n).round(2),
        "country_risk": rng.uniform(1,10,size=n).round(2),
        "maturity_years": rng.uniform(1,30,size=n).round(1),
        "restructured_flag": rng.choice([0,1],size=n,p=[0.92,0.08]),
    })
    pd_s = df["pd_score"]; lgd = df["loss_given_default"]; coll = np.clip(df["collateral_ratio"],0,1)
    country = df["country_risk"]/10; rest = df["restructured_flag"]; rating_map={"AAA":0,"AA":1,"A":2,"BBB":3,"BB":4,"B":5,"CCC":6,"D":7}
    rating_num = df["credit_rating"].map(rating_map).values/7
    log_odds = -2.0 + 3.0*pd_s + 0.5*lgd - 0.5*coll + 0.3*country + 0.5*rest + 0.4*rating_num + rng.normal(0,0.4,size=n)
    prob = 1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,80)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(default=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
