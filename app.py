from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import load_taiwan_default, make_synthetic
from src.model import fit_and_evaluate
from src.core import population_stability_index
from src.visualizations import _style, plot_roc_curve, plot_calibration_curve, plot_ks_curve, plot_feature_importance

st.set_page_config(page_title="CreditModel | IFRS-9 Credit Scoring", layout="wide", page_icon="\U0001f3e6")

with st.sidebar:
    st.header("⚙ Config")
    source = st.radio("Data", ["Taiwan credit default (real, 30k)", "Synthetic"])
    tau = st.slider("Threshold", 0.05, 0.95, 0.50, 0.05)
    st.caption("IFRS 9 ECL | monotonic LightGBM")


@st.cache_resource(show_spinner="Fitting monotonic GBM…")
def get_fit(source: str):
    if source.startswith("Taiwan"):
        try:
            data = load_taiwan_default()
        except Exception:
            st.sidebar.warning("OpenML fetch failed — using synthetic data.")
            data = make_synthetic(n=10000)
    else:
        data = make_synthetic(n=10000)
    bundle, metrics = fit_and_evaluate(data)
    return data, bundle, metrics


data, bundle, metrics = get_fit(source)
y_test, proba = bundle["y_test"], bundle["proba_test"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accounts", f"{data['n_samples']:,}")
c2.metric("Default Rate", f"{data['positive_rate']:.1%}")
c3.metric("AUC", f"{metrics['auc_mean']:.4f}")
c4.metric("Gini", f"{metrics['gini']:.4f}")

t1, t2, t3, t4 = st.tabs(["\U0001f4ca Explorer", "\U0001f52c Model Lab", "\U0001f3af IFRS-9 ECL", "\U0001f4c8 Monitoring"])

with t1:
    st.dataframe(data["X"].head(50), use_container_width=True, height=200)
    fig, ax = plt.subplots(figsize=(5, 3)); _style()
    ax.bar(["Good", "Default"], [1 - data["positive_rate"], data["positive_rate"]], color=["#22c55e", "#f43f5e"])
    for i, v in enumerate([1 - data["positive_rate"], data["positive_rate"]]):
        ax.text(i, v + .01, f"{v:.1%}", ha="center", color="white")
    ax.set_title("Default Distribution", color="white"); ax.grid(True, alpha=.2)
    st.pyplot(fig)

with t2:
    st.markdown(f"Monotonic LightGBM — AUC **{metrics['auc_mean']:.4f}** "
                f"(95% CI {metrics['auc_ci'][0]:.4f}–{metrics['auc_ci'][1]:.4f}), "
                f"Gini **{metrics['gini']:.4f}**. Constraints force PD to rise with "
                f"utilization and payment delinquency and fall with payment ratio — "
                f"so score reasons survive a model-risk review.")
    col_a, col_b = st.columns(2)
    with col_a: st.pyplot(plot_roc_curve(y_test, {"Monotonic GBM": proba}))
    with col_b: st.pyplot(plot_calibration_curve(y_test, {"Monotonic GBM": proba}))
    col_c, col_d = st.columns(2)
    with col_c: st.pyplot(plot_ks_curve(y_test, proba))
    with col_d: st.pyplot(plot_feature_importance(bundle["importances"], bundle["features"]))

with t3:
    st.subheader("IFRS 9 Expected Credit Loss (ECL)")
    st.latex(r"\text{ECL} = \text{PD} \times \text{LGD} \times \text{EAD}")
    lgd_val = st.slider("LGD", 0.1, 0.8, 0.45, 0.05)
    if "bill_amt1" in bundle["features"]:
        ead = bundle["X_test"][:, bundle["features"].index("bill_amt1")].clip(min=0)
    else:
        ead = np.full(len(y_test), 10_000.0)
    ecl_12m = proba * lgd_val * ead
    fig, ax = plt.subplots(figsize=(8, 4)); _style()
    ax.hist(ecl_12m, bins=50, color="#22d3ee", alpha=0.6, edgecolor="#1a1f2e")
    ax.axvline(ecl_12m.mean(), color="#f97316", ls="--", lw=2, label=f"Mean ECL=${ecl_12m.mean():,.0f}")
    ax.set_xlabel("ECL ($)"); ax.set_ylabel("Accounts"); ax.set_title("12-Month Expected Credit Loss", color="white")
    ax.legend(); ax.grid(True, alpha=.2)
    st.pyplot(fig)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total ECL (12m)", f"${ecl_12m.sum():,.0f}")
    c2.metric("Avg ECL per Account", f"${ecl_12m.mean():,.0f}")
    c3.metric("ECL / EAD Ratio", f"{(ecl_12m.sum() / max(ead.sum(), 1)):.2%}")
    st.subheader("Stage Allocation (IFRS 9)")
    st.markdown("Stage 1 (performing): PD < 0.10 | Stage 2 (underperforming): PD 0.10–0.30 | Stage 3 (impaired): PD > 0.30")
    stage = pd.cut(proba, bins=[-0.01, 0.10, 0.30, 1.0], labels=["Stage 1", "Stage 2", "Stage 3"])
    stage_counts = pd.Series(stage).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4)); _style()
    ax.pie(stage_counts, labels=stage_counts.index, autopct="%1.1f%%",
           colors=["#22c55e", "#fbbf24", "#f43f5e"], textprops={"color": "white"})
    ax.set_title("IFRS 9 Stage Allocation", color="white")
    st.pyplot(fig)

with t4:
    st.subheader("Model Monitoring")
    st.latex(r"\text{PSI} = \sum_{i} (\text{Actual}_i - \text{Expected}_i) \times \ln\left(\frac{\text{Actual}_i}{\text{Expected}_i}\right)")
    rng = np.random.default_rng(42)
    expected = proba
    actual = np.clip(proba + rng.normal(0, 0.08, size=len(proba)), 0, 1)
    psi_val = population_stability_index(expected, actual)
    fig, ax = plt.subplots(figsize=(8, 4)); _style()
    bins = np.linspace(0, 1, 21)
    ax.hist(expected, bins=bins, alpha=0.5, color="#22d3ee", label="Expected (development)", density=True)
    ax.hist(actual, bins=bins, alpha=0.5, color="#f43f5e", label="Actual (simulated drift)", density=True)
    ax.set_xlabel("PD Score"); ax.set_ylabel("Density")
    ax.set_title(f"Score Drift — PSI={psi_val:.4f}", color="white")
    ax.legend(fontsize=8); ax.grid(True, alpha=.2)
    st.pyplot(fig)
    st.metric("Population Stability Index (PSI)", f"{psi_val:.4f}")
    st.info("PSI < 0.10 = no significant drift. 0.10–0.25 = moderate. > 0.25 = severe (recalibrate).")
