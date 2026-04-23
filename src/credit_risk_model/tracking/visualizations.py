from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import learning_curve


def plot_learning_curve(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    title: str = "Learning Curve",
    train_sizes: np.ndarray | None = None,
) -> tuple[plt.Figure, dict[str, float]]:
    """Plot training vs. cross-validation score as a function of training
    set size.

    Returns both the figure (for mlflow.log_figure) and a metrics dict
    (for mlflow.log_metrics) with final train/val scores and the gap.

    The gap (train_score - val_score) is the key diagnostic:
        gap > 0.15 → likely overfitting (try more regularization or more data)
        val_score low, gap small → likely underfitting (try richer model)
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        shuffle=True,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes_abs, train_mean, "o-", color="steelblue", label="Training score")
    ax.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="steelblue",
    )
    ax.plot(train_sizes_abs, val_mean, "o-", color="darkorange", label="Cross-validation score")
    ax.fill_between(
        train_sizes_abs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="darkorange",
    )
    ax.set_xlabel("Training examples")
    ax.set_ylabel("ROC AUC")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    # Loggable metrics (final training set size)
    lc_metrics = {
        "final_train_score": float(train_mean[-1]),
        "final_val_score": float(val_mean[-1]),
        "bias_variance_gap": float(train_mean[-1] - val_mean[-1]),
    }
    return fig, lc_metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    labels: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot confusion matrix with both counts and percentage annotations."""
    labels = labels or ["Bad (0)", "Good (1)"]
    cm = confusion_matrix(y_true, y_pred)
    n = cm.sum()

    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")

    # Add percentage annotations below counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / n * 100
            ax.text(
                j,
                i + 0.3,
                f"({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    ax.set_title(title)
    ax.grid(False)
    if created_ax:
        plt.tight_layout()
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot PR curve with no-skill baseline and AP annotation."""
    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_proba,
        name="Model",
        ax=ax,
    )

    # No-skill baseline: proportion of positive class
    no_skill = y_true.mean()
    ax.axhline(
        y=no_skill,
        color="gray",
        linestyle="--",
        label=f"No Skill (AP={no_skill:.2f})",
    )

    ap = average_precision_score(y_true, y_proba)
    ax.set_title(f"{title} (AP={ap:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    if created_ax:
        plt.tight_layout()
    return fig
