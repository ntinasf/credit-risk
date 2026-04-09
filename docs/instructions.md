# Credit Risk Model — Instructions

> **Version:** 0.1.0  
> **Python:** ≥ 3.13  
> **Package manager:** [uv](https://docs.astral.sh/uv/) (recommended)  
> **Dataset:** [German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Prerequisites & Installation](#3-prerequisites--installation)
4. [Data Preparation](#4-data-preparation)
5. [Configuration Reference](#5-configuration-reference)
6. [Training Workflow](#6-training-workflow)
7. [Ensemble Scoring](#7-ensemble-scoring)
8. [Exporting Pipelines](#8-exporting-pipelines)
9. [Running the Streamlit App](#9-running-the-streamlit-app)
10. [Docker Deployment](#10-docker-deployment)
11. [CI / CD Pipeline](#11-ci--cd-pipeline)
12. [Testing](#12-testing)
13. [Linting & Formatting](#13-linting--formatting)
14. [MLflow Tracking](#14-mlflow-tracking)
15. [Cost Matrix & Threshold Tuning](#15-cost-matrix--threshold-tuning)
16. [Known Issues & Notes](#16-known-issues--notes)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. Project Overview

This project trains a **4-model soft-voting ensemble** for binary credit-risk
classification on the [German Credit dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).
The four models are:

| Key  | Algorithm                | Encoding strategy                       |
| ---- | ------------------------ | ---------------------------------------- |
| `lrc` | Logistic Regression     | One-hot + WOE + Count + StandardScaler   |
| `rfc` | Random Forest           | One-hot + Ordinal + Target + StandardScaler |
| `svc` | Support Vector Classifier | One-hot + WOE + Target + StandardScaler |
| `cat` | CatBoost                | Native categorical handling (passthrough) |

Each model is trained independently with **Bayesian hyperparameter search**
(`BayesSearchCV` from scikit-optimize), individually cost-tuned thresholds, and
optional SMOTE/SVMSMOTE oversampling. All training runs are tracked in MLflow.

A final ensemble script combines the four pipelines using weighted soft voting
with a configurable approval threshold (default 0.83), registers the ensemble
as a pyfunc model in MLflow, and exports pickle files for the Streamlit app.

Throughout the project, `class = 1` means **good credit / low risk** and
`class = 0` means **bad credit / high risk**. Model probabilities are therefore
interpreted as `P(class=1)`.

---

## 2. Repository Layout

```
credit-risk-project/
├── .github/workflows/
│   └── ci.yml                   # GitHub Actions pipeline (lint → test → Docker)
├── app/
│   ├── data/                    # Sample data for the Streamlit app
│   ├── models/                  # Exported .pkl pipelines (not committed)
│   ├── requirements.txt         # Streamlit app dependencies (for Docker)
│   └── streamlit_app.py         # Streamlit UI
├── data/
│   ├── raw/                     # Original UCI data (german.data + description.txt)
│   └── processed/               # Mapped CSV + train/test splits
├── docs/
│   ├── architecture.html        # Interactive architecture diagram
│   └── instructions.md          # ← you are here
├── scripts/
│   ├── process_data.py          # Map raw german.data → german_credit.csv
│   ├── split_data.py            # Stratified hash split → train + test CSVs
│   ├── export_pipelines.py      # Export MLflow models → pickle files
│   └── score_ensemble.py        # Evaluate ensemble, log to MLflow
├── src/credit_risk_model/
│   ├── config/
│   │   ├── core.py              # Pydantic config loader
│   │   └── model_config.yml     # Single source of truth for all parameters
│   ├── processing/
│   │   ├── catboost_wrapper.py  # Sklearn-compatible CatBoost wrapper
│   │   ├── features.py          # FeatureEngineer (domain transforms)
│   │   └── preprocessors.py     # Pipeline builders (generic + CatBoost)
│   ├── tracking/
│   │   ├── metrics.py           # Cost-aware evaluation metrics
│   │   └── visualizations.py    # Confusion matrix, PR curve, learning curve
│   ├── training/
│   │   ├── base.py              # BaseModelTrainer (Template Method)
│   │   ├── train_catboost.py    # CatBoostTrainer
│   │   ├── train_lrc.py         # LRCTrainer
│   │   ├── train_rf.py          # RFTrainer
│   │   └── train_svc.py         # SVCTrainer
│   ├── ensemble.py              # CreditRiskEnsemble + MLflow pyfunc wrapper
│   └── predict.py               # Unified prediction API
├── tests/
│   ├── conftest.py              # Shared fixtures (data, pipelines)
│   ├── test_features.py         # FeatureEngineer unit + sklearn contract tests
│   ├── test_ensemble.py         # Ensemble integration tests
│   └── test_prediction.py       # End-to-end prediction tests
├── Dockerfile                   # Multi-stage Docker build
├── docker-compose.yml           # MLflow + Streamlit services
├── main.py                      # Entry-point placeholder
└── pyproject.toml               # Build metadata & dependencies
```

---

## 3. Prerequisites & Installation

### 3.1 System requirements

- **Python ≥ 3.13**
- **uv** — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Docker** (optional — for containerised deployment)

### 3.2 Clone & install

```bash
git clone https://github.com/ntinasf/credit-risk.git credit-risk-project
cd credit-risk-project
```

#### Core dependencies only

```bash
uv sync
```

#### With development tools (pytest, ruff, nox)

```bash
uv sync --extra dev
```

#### With Streamlit app

```bash
uv sync --extra app
```

#### Everything

```bash
uv sync --extra dev --extra app
```

> **Note:** `uv sync` creates a `.venv` and installs the package in editable
> mode. There is no need to run a separate `pip install -e .`.

### 3.3 Verify installation

```bash
uv run python -c "from credit_risk_model.config.core import config; print(config.target)"
# → class
```

---

## 4. Data Preparation

The project expects **pre-processed** CSV files in `data/processed/`:

| File               | Description                          |
| ------------------ | ------------------------------------ |
| `train_data.csv`   | Training split (≈ 850 rows)          |
| `test_data.csv`    | Hold-out test split (≈ 150 rows)     |

### 4.1 Expected format

- **Target column:** `class` — binary (0 = good, 1 = bad credit risk)
- **Features:** Raw German Credit columns after initial renaming.
  `FeatureEngineer` (the first pipeline step) handles all domain
  transformations, category consolidation, and derived features.

### 4.2 Placing the data

```bash
# Copy or move your prepared CSVs into the expected location
cp /path/to/your/train_data.csv  data/processed/
cp /path/to/your/test_data.csv   data/processed/
```

The path is resolved by `core.py` as:

```
<project_root>/data/processed/
```

### 4.3 Sample data for the Streamlit app

The Streamlit app can also display a "Random Samples" tab that draws rows from
a sample CSV. Place a copy of `test_data.csv` at:

```
app/data/sample_data.csv
```

---

## 5. Configuration Reference

All parameters live in a single YAML file:

```
src/credit_risk_model/config/model_config.yml
```

The file is validated at import time by Pydantic (`AppConfig`). Any mismatch
between ensemble weight keys and model config keys will raise a
`ValidationError` immediately.

### 5.1 Key parameters

| Section | Key | Default | Meaning |
| --- | --- | --- | --- |
| (root) | `training_data_file` | `train_data.csv` | Training CSV filename |
| (root) | `test_data_file` | `test_data.csv` | Test CSV filename |
| (root) | `target` | `class` | Target column name |
| (root) | `random_state` | `8` | Global random seed |
| (root) | `val_size` | `150` | Hold-out validation size for threshold tuning |
| `cost_matrix` | `false_positive` | `5` | Cost of approving a bad borrower |
| `cost_matrix` | `false_negative` | `1` | Cost of rejecting a good borrower |
| `ensemble` | `threshold` | `0.83` | Default approval threshold for ensemble voting |
| `ensemble.weights` | `lrc / rfc / svc / cat` | `2.5 / 1.5 / 1.5 / 1.0` | Soft-voting weights per model |
| `mlflow` | `backend_store_uri` | `sqlite:///mlflow.db` | MLflow backend store |
| `mlflow` | `experiment_name` | `credit_risk_ensemble` | Experiment name for ensemble runs |

### 5.2 Per-model configuration

Each model in `models:` has:

- `experiment_name` / `registry_name` — MLflow experiment & registered model name
- `cv_folds`, `bayes_n_iter`, `bayes_n_points` — Bayesian search settings
- `use_smote` / `smote_type` — oversampling (`smote` or `svmsmote`)
- Column assignment lists — which columns go to which encoder
  (e.g., `one_hot_cols`, `woe_cols`, `ordinal_cols`, `target_cols`,
  `count_cols`, `numeric_scaled_cols`, `passthrough_cols`)
- CatBoost uses different column keys: `numeric_cols`, `categorical_cols`,
  `passthrough_cols` (because it has its own pipeline builder)

---

## 6. Training Workflow

### 6.1 Architecture overview

Training follows the **Template Method** pattern:

```
BaseModelTrainer  (base.py)
├── get_model_key()      — abstract: returns "lrc", "rfc", etc.
├── get_estimator()      — abstract: returns unfitted estimator
├── get_search_space()   — abstract: returns BayesSearchCV parameter space
├── _build_pipeline()    — hook: builds the sklearn Pipeline (overridden by CatBoost)
└── train()              — shared: data loading → pipeline build → Bayes search → threshold tune → MLflow log
```

### 6.2 Running training

```python
from credit_risk_model.training.train_lrc import LRCTrainer
from credit_risk_model.training.train_rf import RFTrainer
from credit_risk_model.training.train_svc import SVCTrainer
from credit_risk_model.training.train_catboost import CatBoostTrainer

# Train all four models (each logs to its own MLflow experiment)
for Trainer in [LRCTrainer, RFTrainer, SVCTrainer, CatBoostTrainer]:
    trainer = Trainer()
    trainer.train()
```

Or from the command line:

```bash
uv run python -c "
from credit_risk_model.training.train_lrc import LRCTrainer
from credit_risk_model.training.train_rf import RFTrainer
from credit_risk_model.training.train_svc import SVCTrainer
from credit_risk_model.training.train_catboost import CatBoostTrainer

for T in [LRCTrainer, RFTrainer, SVCTrainer, CatBoostTrainer]:
    T().train()
"
```

### 6.3 What happens during `.train()`

1. Load `train_data.csv` and split off a validation set (`val_size` rows)
2. Apply `FeatureEngineer` to generate derived features
3. Build the full sklearn `Pipeline` via `_build_pipeline()`:
   - LRC / RFC / SVC → `build_pipeline()` (generic encoder-based)
   - CatBoost → `build_catboost_pipeline()` (passthrough + native categoricals)
4. Run `BayesSearchCV` with cost-aware cross-validation
5. Tune the decision threshold on the validation set (cost-matrix optimised)
6. Log hyperparams, metrics, plots, and the fitted pipeline to MLflow
7. Register the pipeline under the model's `registry_name`

---

## 7. Ensemble Scoring

After all four models are trained and registered in MLflow:

```bash
uv run python scripts/score_ensemble.py
```

Options:

```bash
# Override the decision threshold (default is 0.83 from config)
uv run python scripts/score_ensemble.py --threshold 0.55
```

This script:

1. Loads the four latest pipelines from the MLflow model registry
2. Builds a `CreditRiskEnsemble` (weighted soft voting)
3. Evaluates on `test_data.csv`
4. Logs metrics, confusion matrix, and PR curve to the ensemble experiment
5. Registers the ensemble as a pyfunc model (`credit-risk-ensemble`)

---

## 8. Exporting Pipelines

The Streamlit app and Docker container load models from **pickle files**, not
the MLflow registry. Use the export script to bridge the two:

```bash
uv run python scripts/export_pipelines.py
```

This:

1. Loads the four latest pipelines from MLflow
2. Saves them as `lrc_pipeline.pkl`, `rfc_pipeline.pkl`, `svc_pipeline.pkl`,
   `cat_pipeline.pkl` in `app/models/`

### Options

```bash
# Export to a custom directory
uv run python scripts/export_pipelines.py --output-dir /path/to/models

# Export specific versions
uv run python scripts/export_pipelines.py --versions lrc=3 rfc=2 svc=1 cat=latest
```

### Also copy sample data

```bash
cp data/processed/test_data.csv app/data/sample_data.csv
```

---

## 9. Running the Streamlit App

### 9.1 Prerequisites

Before running the app, make sure:

- [ ] Pipeline pickle files exist in `app/models/`
  (see [§8 Exporting Pipelines](#8-exporting-pipelines))
- [ ] (Optional) Sample data exists in `app/data/sample_data.csv`
  for the "Random Samples" tab
- [ ] The `app` extra is installed: `uv sync --extra app`

### 9.2 Launch

```bash
uv run streamlit run app/streamlit_app.py
```

The app opens at <http://localhost:8501> and provides two tabs:

| Tab | Description |
| --- | --- |
| **Random Samples** | Draw random rows from `sample_data.csv`, show predictions with true labels |
| **Manual Input** | Fill in a form for a single applicant and get a live prediction |

### 9.3 Features

- Individual model probabilities + ensemble weighted average
- Cost-aware verdict: "Low Risk" / "High Risk" based on the approval threshold
- Expected-cost breakdown per applicant
- Feature descriptions and help text for every input
- Progress bar during model loading

---

## 10. Docker Deployment

### 10.1 Dockerfile

The `Dockerfile` uses a two-stage build:

1. **Builder stage** — installs all dependencies via `uv sync --no-dev`
2. **Runtime stage** — copies only `.venv` and source code, runs Streamlit

```bash
docker build -t credit-risk-app .
```

> **Important:** Before building, ensure `app/models/` contains the exported
> `.pkl` files and `app/data/` has `sample_data.csv`. These are copied into
> the image during build.

### 10.2 docker-compose

```bash
docker compose up -d
```

This starts two services:

| Service  | Port  | Description                                    |
| -------- | ----- | ---------------------------------------------- |
| `mlflow` | 5000  | MLflow tracking server (SQLite backend)        |
| `app`    | 8501  | Streamlit credit risk app                      |

The compose file mounts `app/models/` and `app/data/` as **read-only volumes**,
so you can update pickles without rebuilding the image:

```bash
# Re-export after retraining
uv run python scripts/export_pipelines.py
# Restart just the app container
docker compose restart app
```

### 10.3 Environment variable

The compose file sets `CREDIT_RISK_MLFLOW_TRACKING_URI=http://mlflow:5000`,
allowing the app container to reach the MLflow service for registry-based
loading (as an alternative to pickle files).

---

## 11. CI / CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) triggers on pushes
and PRs to `main`. It has **three sequential jobs**:

### 11.1 Job: `lint`

```
Runs on: ubuntu-latest
```

1. **Checks out** the repository
2. **Installs uv** via `astral-sh/setup-uv@v3`
3. **Installs Python 3.13** via `uv python install 3.13`
4. **Installs ruff** via `uv tool install ruff`
5. **Lints** with `uv run ruff check src/ tests/ app/`
6. **Format checks** with `uv run ruff format --check src/ tests/ app/`

### 11.2 Job: `test`

```
Runs on: ubuntu-latest
Depends on: lint
```

1. Checks out & installs uv
2. **Syncs all dev dependencies**: `uv sync --extra dev`
3. **Data preprocessing** — runs `scripts/process_data.py` (maps raw UCI data)
   and `scripts/split_data.py` (stratified train/test split)
4. **Runs pytest** with coverage:
   `uv run pytest tests/ -v --cov=credit_risk_model --cov-report=xml`
5. **Uploads coverage** to Codecov (requires `CODECOV_TOKEN` secret)

### 11.3 Job: `build-docker`

```
Runs on: ubuntu-latest
Depends on: test
Condition: only on main branch pushes
```

1. Checks out the repository
2. **Builds the Docker image**: `docker build -t credit-risk-app:<sha> .`
3. **Smoke test** (conditional) — if `app/models/` contains `.pkl` files,
   starts the container, waits 10 seconds, and hits the Streamlit health
   endpoint (`/_stcore/health`). Skips gracefully when no models exist.

### 11.4 Required repository secrets

| Secret          | Required by    | Notes                              |
| --------------- | -------------- | ---------------------------------- |
| `CODECOV_TOKEN` | `test` job     | Obtain from [codecov.io](https://codecov.io) |

---

## 12. Testing

```bash
# Run all tests (25 tests)
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ -v --cov=credit_risk_model --cov-report=term-missing

# Run a specific test file
uv run pytest tests/test_features.py -v
```

`pyproject.toml` sets `testpaths = ["tests"]` so `pytest` discovers tests
automatically.

### 12.1 Test files

| File | Tests | What it covers |
| --- | --- | --- |
| `conftest.py` | — | Shared fixtures: `raw_sample`, `X_train_small` / `y_train_small` (stratified), `mock_pipelines` (all 4 models fitted) |
| `test_features.py` | 14 | FeatureEngineer transforms: column drops, binary flags, log transforms, category consolidation, binning, duplicate flags, immutability, sklearn contract (clone, get/set_params, repr) |
| `test_ensemble.py` | 6 | Ensemble integration: probability ranges, binary predictions, weight influence, threshold effects, threshold optimisation |
| `test_prediction.py` | 5 | End-to-end `make_prediction()` API: return structure, binary output, probability ranges, model breakdown, error handling |

### 12.2 Test fixture notes

The `X_train_small` / `y_train_small` fixtures use **stratified sampling**
(100 rows per class) rather than a simple `.head()` slice. This is necessary
because the hash-based split in `split_data.py` can group all rows of one
class together at the top of the file. The WOE encoder raises `ValueError` if
the target contains only one class.

---

## 13. Linting & Formatting

This project uses **ruff** (configured in `pyproject.toml`):

```toml
[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

```bash
# Check for lint errors
uv run ruff check src/ tests/ app/

# Auto-fix lint errors
uv run ruff check --fix src/ tests/ app/

# Check formatting
uv run ruff format --check src/ tests/ app/

# Auto-format
uv run ruff format src/ tests/ app/
```

---

## 14. MLflow Tracking

### 14.1 Default backend

By default, MLflow logs to a **local SQLite database** (`sqlite:///mlflow.db`
in the project root). Artifacts are stored under `./mlruns/`.

### 14.2 Viewing the UI

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open <http://localhost:5000>.

### 14.3 Via docker-compose

```bash
docker compose up mlflow -d
```

The MLflow UI is available at <http://localhost:5000>. The `mlflow.db` file
and `mlruns/` directory are bind-mounted from the project root.

### 14.4 Experiments

| Experiment name                   | Created by                   |
| --------------------------------- | ---------------------------- |
| `credit_risk_logistic_regression` | `LRCTrainer`                 |
| `credit_risk_random_forest`       | `RFTrainer`                  |
| `credit_risk_svc`                 | `SVCTrainer`                 |
| `credit_risk_catboost`            | `CatBoostTrainer`            |
| `credit_risk_ensemble`            | `score_ensemble.py`          |

---

## 15. Cost Matrix & Threshold Tuning

### 15.1 Cost model

The German Credit dataset documentation defines an asymmetric cost:

| Decision | True label | Cost |
| --- | --- | --- |
| Approve (predict 1) | Bad (0) | **5** |
| Reject (predict 0) | Good (1) | **1** |

> In this project, `class = 1` means **good credit / low risk**, so:

> - `false_positive = 5` → approving a bad borrower (costly)
> - `false_negative = 1` → rejecting a good borrower (opportunity cost)

### 15.2 Threshold tuning

During training, each model's decision threshold is tuned on a held-out
validation set to minimise the expected cost:

```
expected_cost = FP × cost_FP + FN × cost_FN
```

The ensemble threshold (default 0.83) is set in `model_config.yml` and can be
overridden when running `score_ensemble.py --threshold <value>`.

---

## 16. Known Issues & Notes

### 16.1 CI: Codecov requires a repository secret

The coverage upload step uses `codecov/codecov-action@v4`, which requires a
`CODECOV_TOKEN` secret to be configured in your GitHub repository settings.
Without it, the upload will fail (non-fatal — tests still pass).

### 16.2 CatBoost pipeline builder

The CatBoost trainer uses a dedicated `build_catboost_pipeline()` function
(via the `_build_pipeline()` hook override) because CatBoost handles
categorical features natively instead of through external encoders. The
`estimator` parameter passed to the hook is ignored — `build_catboost_pipeline`
creates its own `CatBoostSklearnWrapper` internally.

### 16.3 Feature value matching

Three sources must agree on categorical string values:

1. **`scripts/process_data.py`** — maps UCI codes to human-readable labels
2. **`processing/features.py`** (`FeatureEngineer`) — uses set-membership
   checks for category consolidation
3. **`app/streamlit_app.py`** — dropdown values in the manual input form

If any of these use different strings, `FeatureEngineer` won’t consolidate
categories correctly and models will receive unexpected levels.

Examples of values that must match exactly:

- `savings_account_bonds`: `"100 - 500 DM"` (not `"100 <= ... < 500 DM"`)
- `credit_history`: `"no credits taken/all credits paid back duly"`
  (not `"no credits/all paid duly"`)

### 16.4 CatBoostSklearnWrapper and sklearn 1.8+

The `CatBoostSklearnWrapper` class implements `__sklearn_tags__()` by
delegating to `super().__sklearn_tags__()`. Older implementations that
called `Tags()` directly will break on sklearn ≥ 1.8, which requires
`estimator_type` and `target_tags` as positional arguments.

Similarly, `FeatureEngineer` and `BaselineEngineer` must inherit
`TransformerMixin` **before** `BaseEstimator` to ensure sklearn's
tag-resolution machinery works correctly:

```python
# ✅ Correct (sklearn ≥ 1.6)
class FeatureEngineer(TransformerMixin, BaseEstimator): ...

# ❌ Wrong — causes RuntimeError during tag resolution
class FeatureEngineer(BaseEstimator, TransformerMixin): ...
```

---

## 17. Troubleshooting

### `ModuleNotFoundError: No module named 'credit_risk_model'`

The package isn't installed in the active environment.

```bash
uv sync            # creates .venv and installs in editable mode
uv run python ...  # ensures the right venv is used
```

### `FileNotFoundError: ... train_data.csv`

Run the data preparation pipeline to generate the training files:

```bash
uv run python scripts/process_data.py
uv run python scripts/split_data.py --test-size 0.15
```

See [§4 Data Preparation](#4-data-preparation).

### `RuntimeError: Failed to load model 'lrc' from registry`

The model hasn't been trained yet. Run training first:

```bash
uv run python -c "from credit_risk_model.training.train_lrc import LRCTrainer; LRCTrainer().train()"
```

### `FileNotFoundError: Pipeline file not found: .../lrc_pipeline.pkl`

Run the export script to create pickle files from MLflow:

```bash
uv run python scripts/export_pipelines.py
```

### Streamlit app shows "No models found"

Ensure `app/models/` contains all four `.pkl` files. See [§8](#8-exporting-pipelines).

### Docker container crashes immediately

Check `docker logs <container>` — most likely `app/models/` is empty.
Export pipelines first, then rebuild or restart:

```bash
uv run python scripts/export_pipelines.py
docker compose restart app
```

### MLflow UI shows "No experiments"

Ensure the tracking URI matches. By default, models log to
`sqlite:///mlflow.db` (relative to your working directory).
Start the UI from the project root:

```bash
cd /path/to/credit-risk-project
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### ruff reports line-length errors

The project uses `line-length = 100` in `pyproject.toml`. If your editor
reports 79-character limits, ensure it's reading the project's ruff config,
not a global one. Run ruff directly to verify:

```bash
uv run ruff check --select E501 src/
```
