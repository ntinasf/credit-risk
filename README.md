# Credit Risk Classification

An end-to-end machine learning project for **credit approval decisions** on the
[German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).
The system trains four complementary models, combines them in a weighted
ensemble, tracks experiments with MLflow, exports deployable inference
artifacts, and serves predictions through a Streamlit app.

**Hosted demo:** [Try here](https://credit-risk-h6wzqyepauzgpp29kypx9e.streamlit.app)
**Repository:** [https://github.com/ntinasf/credit-risk](https://github.com/ntinasf/credit-risk)

---

## Overview

This repository is built around a practical lending problem: how to make better
approval decisions when the cost of a bad approval is much higher than the cost
of a missed good customer.

The project includes:

- raw data processing and train/test preparation
- custom feature engineering
- model-specific preprocessing pipelines
- Bayesian hyperparameter search
- cost-sensitive threshold optimization
- weighted ensemble scoring
- MLflow experiment tracking and registry integration
- exported pickle pipelines for app inference
- a Streamlit application for interactive prediction
- Docker support and automated tests

---

## Highlights

| Area | What it demonstrates |
| --- | --- |
| **Business Framing** | Cost-sensitive decisioning with `FP = 5` and `FN = 1` |
| **Modeling** | Four-model weighted soft-voting ensemble |
| **Feature Engineering** | Reusable domain-specific `FeatureEngineer` transformer |
| **Preprocessing** | Model-specific encoders instead of one generic pipeline |
| **Imbalance Handling** | SMOTE, SVMSMOTE, and cost-aware weighting strategies |
| **Tracking** | MLflow runs, artifacts, and model registry |
| **Inference Delivery** | Exported pickle pipelines + Streamlit app |
| **Quality** | pytest coverage for features, predictions, ensemble logic, and semantics |
| **Deployment** | Docker and `docker compose` support |

---

## Dataset and Target Semantics

The German Credit Dataset contains **1,000 applicants** and **20 features**.

This project uses an explicit target definition that is important to understand:

| Encoded class | Meaning |
| --- | --- |
| `1` | Good credit / low risk |
| `0` | Bad credit / high risk |

That is the reverse of many common credit-risk conventions, where `1` often
means default or bad risk. In this repository:

```text
P(class = 1) = probability of good credit
```

The Streamlit app also shows:

```text
Default risk = 1 - P(class = 1)
```

This split is intentional: the modeling layer stays consistent with the encoded
target, while the UI presents a more intuitive business-facing risk view.

---

## The Ensemble

The final ensemble combines four trained models:

| Key | Model | Weight |
| --- | --- | ---: |
| `lrc` | Logistic Regression | 2.5 |
| `rfc` | Random Forest | 1.5 |
| `svc` | Support Vector Classifier | 1.5 |
| `cat` | CatBoost | 1.0 |

Ensemble probability:

```text
P_ensemble(class = 1) =
(2.5 * P_lrc + 1.5 * P_rfc + 1.5 * P_svc + 1.0 * P_cat) / 6.5
```

Decision rule:

- if `P(class = 1) >= 0.83` → **Low Risk**
- otherwise → **High Risk**

Equivalent app-facing interpretation:

- if `P(default) <= 0.17` → **Low Risk**
- otherwise → **High Risk**

---

## Cost-Sensitive Objective

The project is tuned for an asymmetric cost structure:

| Error Type | Meaning | Cost |
| --- | --- | ---: |
| False Positive | Approve a bad borrower | 5 |
| False Negative | Reject a good borrower | 1 |

This cost model drives threshold selection and evaluation, making the system
more conservative than a plain accuracy-optimized classifier.

---

## Modeling Approach

### Base models

The ensemble deliberately mixes four different learning styles:

- **Logistic Regression** for a strong linear baseline
- **Random Forest** for nonlinear interactions
- **Support Vector Classifier** for margin-based separation
- **CatBoost** for boosted tree performance with native categorical handling

### Feature engineering

The custom `FeatureEngineer` adds domain-informed features such as:

- monthly burden features
- duration-to-age relationships
- age-group bins
- transformed credit amount features
- sparse-category consolidation
- no-checking-account signals

### Model-specific preprocessing

The preprocessing stack is not one-size-fits-all. Depending on the model, the
project uses combinations of:

- one-hot encoding
- weight of evidence encoding
- target encoding
- count encoding
- ordinal encoding
- CatBoost native categorical handling

### Imbalance handling

Different models use different strategies, including:

- **SMOTE**
- **SVMSMOTE**
- cost-sensitive weighting

---

## Architecture

```text
credit-risk-project/
├── app/
│   ├── data/
│   │   └── sample_data.csv
│   ├── models/
│   └── streamlit_app.py
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── architecture.html
│   └── instructions.md
├── notebooks/
│   ├── example.ipynb
│   └── investigation.ipynb
├── scripts/
│   ├── export_pipelines.py
│   ├── process_data.py
│   ├── score_ensemble.py
│   └── split_data.py
├── src/credit_risk_model/
│   ├── config/
│   ├── processing/
│   ├── tracking/
│   ├── training/
│   ├── ensemble.py
│   ├── predict.py
│   └── target_semantics.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── main.py
├── pyproject.toml
└── README.md
```

---

## Streamlit App

The app is built for both demos and inspection.

It supports:

- random applicant samples from the dataset
- full manual applicant input
- ensemble probability output
- per-model probability breakdown
- explicit **Good Credit Probability** and **Default Risk** display
- final **Low Risk / High Risk** verdict based on the production threshold

Before launching the app locally, export the trained pipelines to
`app/models/`.

---

## Quickstart

### Prerequisites

- Python `>= 3.13`
- [`uv`](https://docs.astral.sh/uv/)
- Docker optional

### Clone and install

```bash
git clone https://github.com/ntinasf/credit-risk.git credit-risk-project
cd credit-risk-project
uv sync --extra dev --extra app
```

### Prepare the data

```bash
uv run python scripts/process_data.py
uv run python scripts/split_data.py
```

### Train models

Train all four models:

```bash
uv run python main.py
```

Train without Bayesian tuning:

```bash
uv run python main.py --no-tune
```

Train a single model:

```bash
uv run python main.py --model lrc
uv run python main.py --model rfc
uv run python main.py --model svc
uv run python main.py --model cat
```

### Score the ensemble and export app artifacts

```bash
uv run python scripts/score_ensemble.py
uv run python scripts/export_pipelines.py
```

### Launch the Streamlit app

```bash
uv run streamlit run app/streamlit_app.py
```

### Run tests

```bash
uv run pytest tests/ -v
```

---

## Docker

Run the app and MLflow locally with Docker Compose:

```bash
docker compose up --build
```

Default local services:

- **MLflow:** `http://localhost:5000`
- **Streamlit:** `http://localhost:8501`

---

## Documentation

- [`docs/instructions.md`](docs/instructions.md) — setup, configuration,
  workflow, and troubleshooting
- [`docs/architecture.html`](docs/architecture.html) — system overview and
  architecture diagram
- `notebooks/investigation.ipynb` — analysis and evaluation workflow

