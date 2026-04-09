from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class CatBoostSklearnWrapper(ClassifierMixin, BaseEstimator):
    """CatBoostClassifier with full sklearn interface compliance.

    This wrapper:
    1. Implements __sklearn_tags__() so sklearn.clone() works correctly
    2. Exposes get_params() / set_params() for BayesSearchCV
    3. Implements predict_proba() in the format sklearn expects
       (n_samples, n_classes)
    4. Stores cat_features so the Pipeline can pass them through correctly

    All CatBoost constructor parameters are forwarded via **kwargs. This means
    BayesSearchCV can tune any CatBoost parameter using the standard 'model__param'
    prefix — no special-casing required.
    """

    def __init__(
        self,
        depth: int = 6,
        learning_rate: float = 0.1,
        iterations: int = 500,
        l2_leaf_reg: float = 3.0,
        border_count: int = 128,
        scale_pos_weight: float = 5.0,
        random_seed: int = 8,
        verbose: int = 0,
        cat_features: list[int] | None = None,
        task_type: str = "CPU",
        thread_count: int = -1,
    ) -> None:
        # Store all parameters as instance attributes with the SAME names
        # as __init__ arguments. Required for BaseEstimator.get_params() to work.
        self.depth = depth
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.scale_pos_weight = scale_pos_weight
        self.random_seed = random_seed
        self.verbose = verbose
        self.cat_features = cat_features
        self.task_type = task_type
        self.thread_count = thread_count

    def _make_catboost(self) -> CatBoostClassifier:
        """Construct CatBoostClassifier from current parameters."""
        return CatBoostClassifier(
            depth=self.depth,
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            l2_leaf_reg=self.l2_leaf_reg,
            border_count=self.border_count,
            scale_pos_weight=self.scale_pos_weight,
            random_seed=self.random_seed,
            verbose=self.verbose,
            cat_features=self.cat_features,
            task_type=self.task_type,
            thread_count=self.thread_count,
        )

    def fit(self, X, y, **fit_params):
        self.model_ = self._make_catboost()
        self.model_.fit(X, y, **fit_params)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        return self.model_.predict(X).flatten()

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        # CatBoost returns shape (n_samples, n_classes) — consistent with sklearn
        return self.model_.predict_proba(X)

    def __sklearn_tags__(self):
        """Provide sklearn tags so clone() and BayesSearchCV work correctly."""
        tags = super().__sklearn_tags__()
        return tags
