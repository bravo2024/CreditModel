from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import make_synthetic
from src.model import train_all_models, cross_validate
from src.core import compute_metrics
from src.visualizations import *
st.set_page_config(page_title="CreditModel | Aptivaa IFRS-9", layout="wide", page_icon="\U0001f3e6")
with st.sidebar:
    st.header("\u2699 Config"); n=st.slider("Samples",2000,20000,10000,1000)
    tau=st.slider("Threshold",0.05,0.95,0.50,0.05)
    st.caption("Aptivaa | IFRS 9 ECL | Basel III | SR 11-7")
data=make_synthetic(n=n); b=train_all_models(data)
y_test=b["y_test"]; y_probas={n:b["results"][n]["y_proba"] for n in b["results"]}
best=max(b["results"],key=lambda n: b["results"][n]["metrics"].get("roc_auc",0))
c1,c2,c3,c4=st.columns(4)
c1.metric("Samples",f"{n:,}"); c2.metric("Default Rate",f"{data['positive_rate']:.1%}")
c3.metric("Best AUC",f"{b['results'][best]['metrics']['roc_auc']:.4f}"); c4.metric("Best",best)
t1,t2,t3,t4=st.tabs(["\U0001f4ca Explorer","\U0001f52c Model Lab","\U0001f3af IFRS-9 ECL","\U0001f4ca Monitoring"])
with t1:
    st.dataframe(data["df"].head(50),use_container_width=True,height=200)
    fig,ax=plt.subplots(figsize=(5,3)); _style()
    ax.bar(["Good","Default"],[1-data["positive_rate"],data["positive_rate"]],color=["#22c55e","#f43f5e"])
    for i,v in enumerate([1-data["positive_rate"],data["positive_rate"]]): ax.text(i,v+.01,f"{v:.1%}",ha="center",color="white")
    ax.set_title("Default Distribution",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
with t2:
    rows=[{**{"Model":n},**{k:f"{v:.4f}" for k,v in r["metrics"].items() if k!="confusion_matrix"}} for n,r in b["results"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
    col_a,col_b=st.columns(2)
    with col_a: st.pyplot(plot_roc_curve(y_test,y_probas))
    with col_b: st.pyplot(plot_calibration_curve(y_test,y_probas))
    st.pyplot(plot_confusion_matrix(y_test,b["results"]["XGBoost"]["y_pred"],"XGBoost"))
    cv=cross_validate(data)
    cvr=[{"Model":n,"AUC":f"{s['roc_auc']['mean']:.4f}","\u00b1":f"\u00b1{s['roc_auc']['std']:.4f}"} for n,s in cv.items()]
    st.dataframe(pd.DataFrame(cvr).set_index("Model"),use_container_width=True)
with t3:
    st.subheader("IFRS 9 Expected Credit Loss (ECL)")
    st.latex(r"\text{ECL} = \sum \text{PD}_t \times \text{LGD}_t \times \text{EAD}_t \times \text{DF}_t")
    xgb_y=b["results"]["XGBoost"]["y_proba"]
    lgd_val=st.slider("LGD",0.1,0.8,0.45,0.05); ead=data["df"]["exposure_at_default"].values
    ecl_12m = xgb_y * lgd_val * ead
    fig,ax=plt.subplots(figsize=(8,4)); _style()
    ax.hist(ecl_12m, bins=50, color="#22d3ee", alpha=0.6, edgecolor="#1a1f2e")
    ax.axvline(ecl_12m.mean(), color="#f97316", ls="--", lw=2, label=f"Mean ECL=${ecl_12m.mean():,.0f}")
    ax.set_xlabel("ECL ($)"); ax.set_ylabel("Count"); ax.set_title("12-Month Expected Credit Loss",color="white")
    ax.legend(); ax.grid(True,alpha=.2)
    st.pyplot(fig)
    c1,c2,c3=st.columns(3)
    c1.metric("Total ECL (12m)",f"${ecl_12m.sum():,.0f}")
    c2.metric("Avg ECL per Loan",f"${ecl_12m.mean():,.0f}")
    c3.metric("ECL / EAD Ratio",f"{(ecl_12m.sum()/ead.sum()):.2%}")
    st.subheader("Stage Allocation (IFRS 9)")
    st.markdown("Stage 1 (performing): PD < 0.10 | Stage 2 (underperforming): PD 0.10-0.30 | Stage 3 (impaired): PD > 0.30")
    stage = pd.cut(xgb_y, bins=[-0.01,0.10,0.30,1.0], labels=["Stage 1","Stage 2","Stage 3"])
    stage_counts = pd.Series(stage).value_counts()
    fig,ax=plt.subplots(figsize=(6,4)); _style()
    colors=["#22c55e","#fbbf24","#f43f5e"]
    ax.pie(stage_counts, labels=stage_counts.index, autopct="%1.1f%%", colors=colors, textprops={"color":"white"})
    ax.set_title("IFRS 9 Stage Allocation",color="white")
    st.pyplot(fig)
with t4:
    st.subheader("Model Monitoring")
    st.latex(r"\text{PSI} = \sum_{i} (\text{Actual}_i - \text{Expected}_i) \times \ln\left(\frac{\text{Actual}_i}{\text{Expected}_i}\right)")
    from src.core import population_stability_index as psi_fn
    expected=xgb_y; rng=np.random.default_rng(42); actual=np.clip(xgb_y+rng.normal(0,0.08,size=len(xgb_y)),0,1)
    psi_val=psi_fn(expected,actual)
    fig,ax=plt.subplots(figsize=(8,4)); _style()
    bins=np.linspace(0,1,21)
    ax.hist(expected,bins=bins,alpha=0.5,color="#22d3ee",label="Expected (development)",density=True)
    ax.hist(actual,bins=bins,alpha=0.5,color="#f43f5e",label="Actual (current)",density=True)
    ax.set_xlabel("PD Score"); ax.set_ylabel("Density")
    ax.set_title(f"Score Drift — PSI={psi_val:.4f}",color="white")
    ax.legend(fontsize=8); ax.grid(True,alpha=.2)
    st.pyplot(fig)
    st.metric("Population Stability Index (PSI)",f"{psi_val:.4f}")
    st.info("PSI < 0.10 = no significant drift. 0.10-0.25 = moderate drift. > 0.25 = severe drift (model may need recalibration).")
