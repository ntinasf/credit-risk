from __future__ import annotations

import logging

import mlflow
import numpy as np
import pandas as pd

from credit_risk_model.config.core import AppConfig, config
from credit_risk_model.tracking.metrics import calculate_cost, evaluate_model

logger = logging.getLogger(__name__)


class CreditRiskEnsemble:
    """Soft-voting ensemble of four credit risk classifiers.

    Each sub-model is a full sklearn Pipeline (FeatureEngineer + encoding
    + classifier).
    The ensemble combines their probability outputs via a weighted average,
    then applies a cost-optimized approval threshold.

    Parameters
    ----------
    pipelines : dict mapping model key → fitted sklearn Pipeline
        e.g., {"lrc": lrc_pipeline, "rfc": rfc_pipeline, ...}
    weights : dict mapping model key → float weight
        Loaded from config.ensemble.weights by default.
    threshold : float
        Decision boundary for predict(). Loaded from config by default.
    app_config : AppConfig
        The application configuration (for cost matrix access).
    """

    def __init__(
        self,
        pipelines: dict,
        weights: dict[str, float] | None = None,
        threshold: float | None = None,
        app_config: AppConfig = config,
    ) -> None:
        self.pipelines = pipelines
        self.weights = weights or dict(app_config.ensemble.weights)
        self.threshold = threshold if threshold is not None else app_config.ensemble.threshold
        self.app_config = app_config

        # Validate alignment between pipelines and weights
        if set(pipelines.keys()) != set(self.weights.keys()):
            raise ValueError(
                f"Pipeline keys {set(pipelines.keys())} "
                f"don't match weight keys {set(self.weights.keys())}"
            )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return weighted average P(class=1 = good borrower)."""
        weighted_sum = np.zeros(len(X))
        total_weight = sum(self.weights.values())

        for model_key, pipeline in self.pipelines.items():
            weight = self.weights[model_key]
            proba = pipeline.predict_proba(X)[:, 1]
            weighted_sum += weight * proba

        return weighted_sum / total_weight

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions where 1=good and 0=bad."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def predict_with_breakdown(self, X: pd.DataFrame) -> dict:
        """Return ensemble prediction plus per-model probabilities.

        Used by the Streamlit app to show individual model confidence bars.
        """
        result = {}
        for model_key, pipeline in self.pipelines.items():
            result[f"{model_key}_proba"] = pipeline.predict_proba(X)[:, 1]

        ensemble_proba = self.predict_proba(X)
        result["ensemble_proba"] = ensemble_proba
        result["ensemble_pred"] = (ensemble_proba >= self.threshold).astype(int)
        result["threshold"] = self.threshold
        return result

    def optimize_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> float:
        """Grid-search the approval threshold to minimize validation cost.

        Updates self.threshold in place and returns the optimal value.
        """
        proba = self.predict_proba(X_val)
        cost_fp = self.app_config.cost_matrix.false_positive
        cost_fn = self.app_config.cost_matrix.false_negative

        best_threshold = self.threshold
        best_cost = float("inf")

        for threshold in np.arange(0.01, 1.0, 0.01):
            preds = (proba >= threshold).astype(int)
            total_cost, _ = calculate_cost(
                y_val.values,
                preds,
                cost_fp,
                cost_fn,
            )
            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = float(threshold)

        self.threshold = best_threshold
        logger.info(f"Optimal ensemble threshold: {best_threshold:.2f} (cost: {best_cost:.0f})")
        return best_threshold

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Evaluate the ensemble and return a metrics dict."""
        all_metrics = {}

        # Individual model metrics.
        # Each pipeline exposes a standard 2D `predict_proba` output.
        for model_key, pipeline in self.pipelines.items():
            individual_metrics = evaluate_model(
                pipeline,
                X,
                y,
                threshold=0.5,  # individual threshold (not ensemble threshold)
                cost_fp=self.app_config.cost_matrix.false_positive,
                cost_fn=self.app_config.cost_matrix.false_negative,
            )
            for metric_name, value in individual_metrics.items():
                all_metrics[f"{model_key}_{metric_name}"] = value

        # Ensemble metrics (computed directly since predict_proba returns 1D)
        ensemble_proba = self.predict_proba(X)  # 1D array: P(good)
        ensemble_pred = (ensemble_proba >= self.threshold).astype(int)

        total_cost, avg_cost = calculate_cost(
            y.values,
            ensemble_pred,
            cost_fp=self.app_config.cost_matrix.false_positive,
            cost_fn=self.app_config.cost_matrix.false_negative,
        )

        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        ensemble_metrics = {
            "roc_auc": roc_auc_score(y, ensemble_proba),
            "accuracy": accuracy_score(y, ensemble_pred),
            "f1": f1_score(y, ensemble_pred, zero_division=0),
            "precision": precision_score(y, ensemble_pred, zero_division=0),
            "recall": recall_score(y, ensemble_pred, zero_division=0),
            "average_precision": average_precision_score(y, ensemble_proba),
            "cost": total_cost,
            "avg_cost": avg_cost,
        }
        all_metrics.update({f"ensemble_{k}": v for k, v in ensemble_metrics.items()})

        return all_metrics


class CreditRiskPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper for the CreditRiskEnsemble.

    Bridges the custom ensemble class to MLflow's standard predict() interface.
    This is what MLflow serializes when you call mlflow.pyfunc.log_model().

    External consumers load the model with:
        model = mlflow.pyfunc.load_model(
            "models:/credit-risk-ensemble/Production"
        )
        predictions = model.predict(input_df)

    They never need to know about CreditRiskEnsemble or individual pipelines.
    """

    def __init__(self, ensemble: CreditRiskEnsemble | None = None) -> None:
        self.ensemble = ensemble

    def predict(
        self,
        context,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> pd.DataFrame:
        """MLflow pyfunc requires this exact signature."""
        proba = self.ensemble.predict_proba(model_input)
        decisions = self.ensemble.predict(model_input)
        return pd.DataFrame(
            {
                "probability": proba,
                "decision": decisions,
                "threshold": self.ensemble.threshold,
            }
        )
