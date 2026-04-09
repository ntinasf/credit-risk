from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import mlflow
import numpy as np
import pandas as pd
from skopt import BayesSearchCV

from credit_risk_model.config.core import AppConfig, SingleModelConfig, config
from credit_risk_model.processing.preprocessors import build_pipeline
from credit_risk_model.tracking.metrics import (
    calculate_cost,
    evaluate_model,
)
from credit_risk_model.tracking.visualizations import (
    plot_confusion_matrix,
    plot_learning_curve,
    plot_precision_recall_curve,
)

logger = logging.getLogger(__name__)


class BaseModelTrainer(ABC):
    """Base class for all four model trainers.

    Subclasses implement three abstract methods:
    - get_estimator(): return the unfitted sklearn-compatible estimator
    - get_search_space(): return the BayesSearchCV param space dict
    - get_model_key(): return the key in config.models (e.g., "lrc")

    All shared logic (MLflow setup, Bayesian search, threshold tuning,
    evaluation, artifact logging, model registration) lives here.
    """

    def __init__(self, app_config: AppConfig = config) -> None:
        self.app_config = app_config
        self.model_cfg: SingleModelConfig = app_config.models[self.get_model_key()]
        self._pipeline = None  # set after training

    # ── Abstract interface (subclasses must implement) ────────────────────

    @abstractmethod
    def get_model_key(self) -> str:
        """Return the key into config.models, e.g. 'lrc', 'rfc', 'svc', 'cat'."""
        raise NotImplementedError

    @abstractmethod
    def get_estimator(self):
        """Return an unfitted sklearn-compatible estimator."""
        raise NotImplementedError

    @abstractmethod
    def get_search_space(self) -> dict:
        """Return BayesSearchCV parameter space.

        Use Pipeline step names as prefixes:
            'model__C' for a param on the 'model' step
            'smote__k_neighbors' for a param on the 'smote' step
        """
        raise NotImplementedError

    # ── Pipeline construction hook (override for CatBoost) ─────────────

    def _build_pipeline(self, estimator):
        """Build the training pipeline. Override for model-specific pipelines.

        The default implementation uses build_pipeline() which constructs
        FeatureEngineer → ColumnTransformer (with encoders) → [SMOTE] → Estimator.
        CatBoostTrainer overrides this to use build_catboost_pipeline() which
        passes columns through without encoding and injects cat_features indices.
        """
        return build_pipeline(estimator, self.model_cfg)

    # ── Shared training logic ─────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        *,
        tune: bool = True,
        tune_threshold: bool = True,
        log_model: bool = True,
        evaluate: bool = True,
    ):
        """Train the model with optional Bayesian tuning and MLflow tracking.

        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val : held-out validation data (not used in CV)
        tune : if True, run BayesSearchCV; otherwise use defaults
        tune_threshold : if True, optimize decision threshold on X_val
        log_model : if True, register the model in MLflow registry
        evaluate : if True, compute and log validation metrics
        """
        mlflow.set_tracking_uri(self.app_config.mlflow["backend_store_uri"])
        mlflow.set_experiment(self.model_cfg.experiment_name)

        with mlflow.start_run(run_name=self._run_name(tune)) as run:
            # ── Log run configuration ──────────────────────────────────────
            mlflow.log_params(
                {
                    "model_type": self.get_model_key().upper(),
                    "cv_folds": self.model_cfg.cv_folds,
                    "random_state": self.app_config.random_state,
                    "tune_hyperparameters": tune,
                    "use_smote": self.model_cfg.use_smote,
                    "smote_type": self.model_cfg.smote_type if self.model_cfg.use_smote else "none",
                    "n_train_samples": len(X_train),
                    "n_val_samples": len(X_val),
                    "n_features_raw": X_train.shape[1],
                }
            )

            # ── Build pipeline ─────────────────────────────────────────────
            estimator = self.get_estimator()
            pipeline = self._build_pipeline(estimator)

            # ── Fit (with or without hyperparameter tuning) ────────────────
            if tune:
                pipeline = self._run_bayesian_search(pipeline, X_train, y_train)
                mlflow.log_param("bayes_n_iter", self.model_cfg.bayes_n_iter)
            else:
                pipeline.fit(X_train, y_train)
                logger.info(f"[{self.get_model_key()}] Fitted with default parameters")

            # ── Log processed feature count ────────────────────────────────
            try:
                sample_transformed = pipeline[:-1].transform(X_train.head(1))
                mlflow.log_param("n_features_processed", sample_transformed.shape[1])
            except Exception:
                pass  # non-critical

            # ── Threshold tuning ─────────────────────────────────────────────
            threshold = 0.5
            if tune_threshold:
                threshold = self._tune_threshold(pipeline, X_val, y_val)
            mlflow.log_metric("tuned_decision_threshold", threshold)

            self._pipeline = pipeline
            self._threshold = threshold

            # ── Evaluation ─────────────────────────────────────────────────
            if evaluate:
                self._log_evaluation(pipeline, X_train, y_train, X_val, y_val, threshold)

            # ── Model registration ─────────────────────────────────────
            if log_model:
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="model",
                    registered_model_name=self.model_cfg.registry_name,
                    input_example=X_train.head(1),
                )
                logger.info(
                    f"[{self.get_model_key()}] Registered as '{self.model_cfg.registry_name}'"
                )

            logger.info(
                f"[{self.get_model_key()}] Training complete. Run ID: {run.info.run_id[:8]}"
            )

        return pipeline

    def _run_bayesian_search(self, pipeline, X_train, y_train):
        """Run BayesSearchCV on the pipeline and return the best estimator."""
        search_space = self.get_search_space()

        search = BayesSearchCV(
            estimator=pipeline,
            search_spaces=search_space,
            n_iter=self.model_cfg.bayes_n_iter,
            n_points=self.model_cfg.bayes_n_points,
            cv=self.model_cfg.cv_folds,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=self.app_config.random_state,
            refit=True,
        )
        search.fit(X_train, y_train)

        # Log best hyperparameters
        mlflow.log_metric("best_cv_roc_auc", search.best_score_)
        for param_name, param_value in search.best_params_.items():
            # Strip pipeline step prefix for cleaner MLflow param names
            clean_name = param_name.replace("model__", "").replace("smote__", "smote_")
            mlflow.log_param(f"best_{clean_name}", param_value)

        logger.info(f"[{self.get_model_key()}] Best CV ROC AUC: {search.best_score_:.4f}")

        return search.best_estimator_

    def _tune_threshold(
        self,
        pipeline,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> float:
        """Find decision threshold minimizing expected cost on validation set."""
        proba = pipeline.predict_proba(X_val)[:, 1]
        cost_matrix = self.app_config.cost_matrix
        best_threshold = 0.5
        best_cost = float("inf")

        for threshold in np.arange(0.01, 1.0, 0.01):
            predictions = (proba >= threshold).astype(int)
            total_cost, _ = calculate_cost(
                y_val.values,
                predictions,
                cost_fp=cost_matrix.false_positive,
                cost_fn=cost_matrix.false_negative,
            )
            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = threshold

        logger.info(
            f"[{self.get_model_key()}] Optimal threshold: {best_threshold:.2f} "
            f"(cost: {best_cost:.0f})"
        )

        return float(best_threshold)

    def _log_evaluation(self, pipeline, X_train, y_train, X_val, y_val, threshold):
        """Log all evaluation metrics and artifact figures to the active MLflow run."""
        metrics = evaluate_model(
            pipeline,
            X_val,
            y_val,
            threshold=threshold,
            cost_fp=self.app_config.cost_matrix.false_positive,
            cost_fn=self.app_config.cost_matrix.false_negative,
        )
        for name, value in metrics.items():
            mlflow.log_metric(f"val_{name}", value)

        # Learning curve (bias-variance diagnostic)
        # Skip for CatBoost — handled in CatBoostTrainer override
        try:
            fig_lc, lc_metrics = plot_learning_curve(
                pipeline,
                X_train,
                y_train,
                cv=self.model_cfg.cv_folds,
                title=f"{self.get_model_key().upper()} Learning Curve",
            )
            for lc_name, lc_val in lc_metrics.items():
                mlflow.log_metric(f"lc_{lc_name}", lc_val)
            mlflow.log_figure(fig_lc, "learning_curve.png")
        except Exception as exc:
            logger.warning(f"[{self.get_model_key()}] Could not generate learning curve: {exc}")

        # Confusion matrix
        y_pred = (pipeline.predict_proba(X_val)[:, 1] >= threshold).astype(int)
        fig_cm = plot_confusion_matrix(
            y_val.values,
            y_pred,
            title=f"{self.get_model_key().upper()} Confusion Matrix",
        )
        mlflow.log_figure(fig_cm, "confusion_matrix.png")

        # Precision-recall curve
        fig_pr = plot_precision_recall_curve(
            y_val.values,
            pipeline.predict_proba(X_val)[:, 1],
            title=f"{self.get_model_key().upper()} Precision-Recall",
        )
        mlflow.log_figure(fig_pr, "precision_recall_curve.png")

        # MLflow tags for filtering runs in the UI
        mlflow.set_tags(
            {
                "model_family": self.get_model_key().upper(),
                "preprocessing": "FeatureEngineer+Encoders",
                "tuning_method": "BayesSearchCV",
                "imbalance_handling": (
                    self.model_cfg.smote_type.upper() if self.model_cfg.use_smote else "cost_weight"
                ),
            }
        )

    def _run_name(self, tune: bool) -> str:
        key = self.get_model_key().upper()
        smote = f"_{self.model_cfg.smote_type.upper()}" if self.model_cfg.use_smote else ""
        tuned = "_Tuned" if tune else "_Baseline"
        return f"{key}{smote}{tuned}_cv{self.model_cfg.cv_folds}"
