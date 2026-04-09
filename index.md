---
layout: page
title: Credit Risk Classification
subtitle: Cost-Sensitive Ensemble Learning for Credit Approval Decisions
---

When a bank receives a loan application, it has to make a binary decision:
approve or reject. Get it wrong in one direction and you lose a customer. Get
it wrong in the other and you lose money — often a lot more money.

This project builds an end-to-end machine learning system that makes that
decision on the
[German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
(1,000 applicants, 20 features) using a four-model weighted ensemble, and
deploys it as an interactive Streamlit application.

[Try the Live Demo](https://YOUR-STREAMLIT-APP-URL.streamlit.app) ·
[View Repository](https://github.com/ntinasf/credit-risk)

---

## The Problem

Not all classification errors are equal. Approving a borrower who defaults
costs the institution far more than turning away someone who would have repaid.
The German Credit Dataset documentation itself quantifies this with a 5 : 1
cost ratio:

| Error | What happens | Cost |
| --- | --- | ---: |
| False Positive | Approve a bad borrower | 5 |
| False Negative | Reject a good borrower | 1 |

Most tutorials train on this dataset by optimizing accuracy or ROC AUC. This
project takes the cost structure seriously: every model selection, threshold
tuning, and ensemble promotion decision is evaluated against the total
misclassification cost — not just a ranking metric.

---

## Investigation Methodology

Before committing to a final system, I ran a systematic investigation across
**16 experiments** — four models × two feature sets × tuned vs untuned:

| Model | Baseline | Baseline + Tuned | FeatureEngineer | FE + Tuned |
| --- | :---: | :---: | :---: | :---: |
| Logistic Regression | ✓ | ✓ | ✓ | ✓ |
| Random Forest | ✓ | ✓ | ✓ | ✓ |
| SVC | ✓ | ✓ | ✓ | ✓ |
| CatBoost | ✓ | ✓ | ✓ | ✓ |

Each experiment was evaluated on a held-out validation set (n = 150) under the
same cost function. The best configuration per model was then carried forward
into the ensemble stage.

**Key findings from the individual experiments:**

- SVC with the custom `FeatureEngineer` achieved the highest individual
  validation ROC AUC (0.811) and tied for the lowest cost (67).
- Feature engineering improved most models, with the largest gains coming from
  the monthly-burden ratio and the no-checking-account flag.
- Bayesian hyperparameter tuning (`BayesSearchCV`) helped most when paired
  with the richer feature set, particularly for SVC and CatBoost.

---

## The Ensemble

Rather than selecting one winning model, the system combines all four through
weighted soft voting. Each model contributes a calibrated probability of good
credit, and the ensemble averages them:

| Model | Encoding Strategy | Imbalance Handling | Weight |
| --- | --- | --- | ---: |
| Logistic Regression | WOE + Count + One-Hot | SMOTE | 2.5 |
| Random Forest | One-Hot + Ordinal + Target | Cost-sensitive weights | 1.5 |
| SVC | WOE + Target + Count + One-Hot | SVMSMOTE | 1.5 |
| CatBoost | Native categorical | scale_pos_weight = 5 | 1.0 |

The weights and threshold were selected through a structured process:

1. **Weight sweep** — exhaustive grid over all four weights (625 combinations),
   shortlisting the top 5 by ROC AUC and precision.
2. **Threshold tuning** — for each shortlisted candidate, sweep the decision
   boundary from 0.01 to 0.99 on the validation set, selecting the threshold
   that minimizes total cost.
3. **Promotion** — the candidate with the lowest validation cost is promoted to
   production. In this case, that was candidate C4 (weights 2.5 / 1.5 / 1.5 /
   1.0, threshold 0.83).

Threshold tuning alone reduced validation cost by **37.4%** compared to the
default 0.50 boundary — from 123 to 77. That single step had a larger effect
on the business metric than any individual hyperparameter change.

---

## Results

### Hold-out test set (n = 159, completely unseen)

| Model | Total Cost | ROC AUC |
| --- | ---: | ---: |
| Logistic Regression | 80 | 0.794 |
| Random Forest | 81 | 0.803 |
| SVC | 92 | 0.817 |
| CatBoost | 91 | 0.802 |
| **Ensemble** | **78** | **0.807** |

The ensemble achieves the **lowest total cost** on the test set — 2.5% lower
than the best individual model (Logistic Regression). That reduction comes
entirely from the combination of diverse models and the cost-optimized
threshold, not from overfitting to validation data.

### Validation-set ensemble diagnostics (n = 150)

| Metric | Value |
| --- | ---: |
| Precision | 0.925 |
| Recall | 0.462 |
| F1 | 0.616 |
| Average Precision | 0.904 |
| ROC AUC | 0.804 |
| Total Cost | 77 |

The precision/recall trade-off is intentional: the system is tuned to be
**conservative**. A 0.925 precision means that when the model does approve
someone, it is right 92.5% of the time. The lower recall reflects the fact
that the model would rather reject a borderline applicant than risk a costly
false positive — which is exactly what the 5 : 1 cost ratio demands.

---

## How the Decision Works

The target encoding in this project is:

| Class | Meaning |
| --- | --- |
| 1 | Good credit / low risk |
| 0 | Bad credit / high risk |

The ensemble outputs `P(class = 1)` — the probability of good credit. The
production threshold is **0.83**: applicants scoring at or above it are
classified as low risk.

In the Streamlit app, the same value is also shown as:

> **Default Risk = 1 − P(good credit)**

So a threshold of 0.83 on the good-credit probability is equivalent to a
default-risk cutoff of 0.17. Both views are displayed side by side in the app.

---

## Feature Engineering

A custom `FeatureEngineer` (sklearn `TransformerMixin`) adds domain-informed
signals before any encoding:

- **Monthly burden** — `credit_amount / duration`, log-transformed
- **Duration-to-age ratio** — loan length relative to applicant age
- **Age-group bins** — Young, Early Career, Prime, Mature
- **Credit amount bins** — quintile-based categorization with stable bin edges
  learned at training time
- **Sparse-category consolidation** — merging rare levels in job type, housing,
  and other low-frequency categoricals
- **No-checking flag** — binary indicator for applicants without a checking
  account, one of the strongest single predictors in this dataset

Each model then receives its own encoding pipeline. Logistic Regression uses
weight-of-evidence and count encoding; Random Forest uses one-hot and ordinal
encoding; SVC uses a mix of WOE, target, and count encoding; CatBoost uses its
native categorical handler. This model-specific preprocessing is one of the
reasons the ensemble benefits from diversity.

---

## Engineering

This is not a notebook-only project. The full system is structured for
reproducibility and deployment:

- **Config-driven architecture** — a single YAML file + Pydantic v2 validation
  controls every pipeline, encoder, feature list, and search space
- **Template Method training** — one abstract base trainer, four concrete
  subclasses, each configuring its own pipeline and hyperparameter space
- **MLflow integration** — full experiment tracking, artifact logging (confusion
  matrices, learning curves, PR curves), and model registry
- **Pickle export workflow** — registered MLflow models are exported to
  standalone `.pkl` files for lightweight app inference
- **Streamlit app** — interactive UI with manual input, random sampling,
  ensemble and per-model probability breakdowns
- **Docker + Compose** — one-command local deployment of MLflow and Streamlit
- **Automated tests** — pytest suite covering feature engineering, prediction
  contracts, ensemble logic, and target-semantics consistency

---

## Interactive Demo

The Streamlit app lets you inspect the model's behavior, not just run it.

You can:

- load a random applicant from the dataset (good or bad credit)
- enter a full borrower profile manually
- see the ensemble's **good-credit probability** and **default risk**
- compare per-model probability breakdowns
- read the final **Low Risk / High Risk** verdict

[Launch the app →](https://YOUR-STREAMLIT-APP-URL.streamlit.app)

---

## What This Project Demonstrates

- Framing an ML problem around **real business costs**, not just accuracy
- Building a **diverse ensemble** where each model brings a different inductive
  bias and encoding strategy
- Running a **structured model-selection investigation** (16 experiments,
  weight sweep, threshold tuning, promotion)
- Keeping **probability semantics consistent** from training code to UI labels
- Packaging the result as a **testable, deployable application** with MLflow
  tracking, Docker support, and automated tests
