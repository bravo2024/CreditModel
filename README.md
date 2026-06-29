# CreditModel

> IFRS 9 / Basel-compliant credit scoring with multi-model comparison.

Trains and compares four classifiers (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) on synthetic credit data for probability of default (PD) estimation. Includes 5-fold cross-validation, full metric reporting, and scorecard calibration.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.868 |
| Gini | 0.735 |
| KS Statistic | 0.580 |
| F1 Score | 0.603 |
| Accuracy | 0.799 |

5-fold CV AUC: 0.871 ± 0.004. Four models compared.

## Features

| Tab | What it does |
|---|---|
| **Explorer** | Portfolio overview, default distribution, feature correlations |
| **Model Lab** | Multi-model comparison, ROC curves, confusion matrices, CV results |
| **Scoring** | PD score distribution, scorecard bands, approval rate by segment |
| **IFRS 9** | ECL estimation, staging analysis, provision calculation |

## Repo Structure

```
CreditModel/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic credit dataset: income, loan amount, credit history, employment status, DTI ratio, number of accounts, and delinquency history.

## License

MIT
