# CreditModel

A credit scorecard built to IFRS 9 conventions: a monotonically-constrained LightGBM PD model on real revolving-credit data, with ECL staging and drift monitoring on top.

The dataset is the UCI **"default of credit card clients"** study (Yeh & Lien 2009) — 30,000 Taiwanese credit-card accounts with billing/payment history and a one-month default outcome. It's fetched from OpenML on first run and cached under `data/`; a synthetic generator with the same shape covers offline runs (`--synthetic`).

The interesting modeling constraint: PD is **forced monotone** in the features a validator would ask about — more utilization or worse payment status can only push risk up, paying bills down can only push it down. That's what makes a GBM defensible in a model-risk review, at a small cost in raw AUC.

## Results (30k accounts, 25% holdout)

| Metric | Value |
|---|---|
| ROC AUC | 0.772 (95% CI 0.760–0.784) |
| Gini | 0.544 |
| Default rate | 22.1% |

For calibration, KS and feature importances see the Model Lab tab. Numbers regenerate with `python train.py`.

## Getting started

```bash
pip install -r requirements.txt
python train.py              # fetches the UCI data via OpenML once
python train.py --synthetic  # offline check
pytest -q
streamlit run app.py
```

## Dashboard

| Tab | What it does |
|---|---|
| **Explorer** | Portfolio overview, default distribution |
| **Model Lab** | ROC, calibration, KS curve, monotone-GBM feature importances |
| **IFRS-9 ECL** | ECL = PD × LGD × EAD per account (EAD = last statement balance), stage allocation |
| **Monitoring** | PSI score-drift check against a simulated current population |

## Layout

```
src/         Taiwan-default loader + synthetic fallback, monotonic GBM, validation metrics
train.py     training pipeline
app.py       Streamlit dashboard
tests/       smoke tests
data/        cached csv (created on first run)
models/      saved model + metrics (gitignored)
```

MIT licensed.
