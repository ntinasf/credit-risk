from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float = 5.0,
    cost_fn: float = 1.0,
) -> tuple[float, float]:
    """Calculate total and average cost given predictions and a cost matrix.

    Parameters
    ----------
    y_true : ground truth labels (1 = good borrower, 0 = bad borrower)
    y_pred : predicted labels using the same encoding
    cost_fp : cost of predicting good for an actually bad borrower
    cost_fn : cost of predicting bad for an actually good borrower

    Returns
    -------
    (total_cost, avg_cost_per_sample)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    total_cost = fp * cost_fp + fn * cost_fn
    avg_cost = total_cost / len(y_true)

    return float(total_cost), float(avg_cost)


def cost_scorer_fn(estimator, X, y):
    """Negative cost scorer for search routines that expect higher-is-better.

    sklearn scorers follow the convention: higher = better.
    Cost is lower = better, so we negate it.

    This function MUST be importable from the package so that pipelines
    containing TunedThresholdClassifierCV can be deserialized correctly.
    """
    y_pred = estimator.predict(X)
    total_cost, _ = calculate_cost(y, y_pred)
    return -total_cost


def evaluate_model(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
    cost_fp: float = 5.0,
    cost_fn: float = 1.0,
) -> dict[str, float]:
    """Compute all evaluation metrics for a fitted pipeline.

    Returns a dict suitable for mlflow.log_metrics().
    """
    proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    total_cost, avg_cost = calculate_cost(y.values, y_pred, cost_fp, cost_fn)

    return {
        "roc_auc": roc_auc_score(y, proba),
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred, zero_division=0),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "average_precision": average_precision_score(y, proba),
        "cost": total_cost,
        "avg_cost": avg_cost,
    }
