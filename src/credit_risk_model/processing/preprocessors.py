from __future__ import annotations

import category_encoders as ce
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk_model.config.core import SingleModelConfig
from credit_risk_model.processing.features import FeatureEngineer


def build_column_transformer(model_cfg: SingleModelConfig) -> ColumnTransformer:
    """Construct a ColumnTransformer from config.

    The model config specifies which columns get which encoding strategy.
    """
    transformers: list[tuple] = []

    if model_cfg.one_hot_cols:
        transformers.append(
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                model_cfg.one_hot_cols,
            )
        )

    if model_cfg.woe_cols:
        transformers.append(
            (
                "woe",
                ce.WOEEncoder(handle_unknown="value", handle_missing="value"),
                model_cfg.woe_cols,
            )
        )

    if model_cfg.target_cols:
        transformers.append(
            (
                "target",
                ce.TargetEncoder(handle_unknown="value", handle_missing="value"),
                model_cfg.target_cols,
            )
        )

    if model_cfg.count_cols:
        transformers.append(
            (
                "count",
                ce.CountEncoder(handle_unknown="value", handle_missing="value"),
                model_cfg.count_cols,
            )
        )

    if model_cfg.ordinal_cols:
        transformers.append(
            (
                "ordinal",
                ce.OrdinalEncoder(handle_unknown="value", handle_missing="value"),
                model_cfg.ordinal_cols,
            )
        )

    if model_cfg.numeric_scaled_cols:
        transformers.append(
            (
                "scaled",
                StandardScaler(),
                model_cfg.numeric_scaled_cols,
            )
        )

    if model_cfg.passthrough_cols:
        transformers.append(
            (
                "passthrough",
                "passthrough",
                model_cfg.passthrough_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline(
    estimator,
    model_cfg: SingleModelConfig,
) -> Pipeline | ImbPipeline:
    """Wrap estimator + preprocessing into a full sklearn/imblearn Pipeline.

    If use_smote is True in the model config, returns an imblearn Pipeline
    so that SMOTE steps can be included in the pipeline. Otherwise returns
    a standard sklearn Pipeline.

    The FeatureEngineer is always the first step. Its `duplicate_checking`
    and `duplicate_amount` flags are set from the model config, so the
    correct variant of FeatureEngineer is used per model without any
    conditional logic in the calling code.
    """
    feature_engineer = FeatureEngineer(
        duplicate_checking=model_cfg.duplicate_checking,
        duplicate_amount=model_cfg.duplicate_amount,
    )
    preprocessor = build_column_transformer(model_cfg)

    steps = [
        ("feature_engineer", feature_engineer),
        ("preprocessor", preprocessor),
        ("model", estimator),
    ]

    if model_cfg.use_smote:
        from imblearn.over_sampling import SMOTE

        steps.insert(2, ("smote", SMOTE(random_state=8)))
        return ImbPipeline(steps=steps)

    return Pipeline(steps=steps)


def build_catboost_input(
    model_cfg: SingleModelConfig,
) -> tuple[ColumnTransformer, list[int]]:
    """Build a passthrough transformer for CatBoost + return cat_feature indices.

    CatBoost needs to know which column positions are categorical so it can
    apply its internal encoding. This function returns:
    - A ColumnTransformer that selects and reorders columns (no encoding)
    - A list of integer indices indicating which columns are categorical

    The indices are passed to CatBoostClassifier(cat_features=...).
    """
    all_cols = model_cfg.numeric_cols + model_cfg.categorical_cols + model_cfg.passthrough_cols
    # Categorical columns start after all numeric columns
    cat_feature_indices = list(
        range(
            len(model_cfg.numeric_cols),
            len(model_cfg.numeric_cols) + len(model_cfg.categorical_cols),
        )
    )
    selector = ColumnTransformer(
        transformers=[("select", "passthrough", all_cols)],
        remainder="drop",
    )
    return selector, cat_feature_indices


def build_catboost_pipeline(
    model_cfg: SingleModelConfig,
    random_state: int = 8,
    scale_pos_weight: float = 5.0,
) -> Pipeline:
    """Build the full pipeline for CatBoost, including cat_features injection."""
    from credit_risk_model.processing.catboost_wrapper import CatBoostSklearnWrapper

    feature_engineer = FeatureEngineer()  # CatBoost: no duplicate flags needed
    selector, cat_feature_indices = build_catboost_input(model_cfg)
    estimator = CatBoostSklearnWrapper(
        scale_pos_weight=scale_pos_weight,
        random_seed=random_state,
        verbose=0,
        cat_features=cat_feature_indices,  # ← injected at construction time
    )

    return Pipeline(
        steps=[
            ("feature_engineer", feature_engineer),
            ("preprocessor", selector),
            ("model", estimator),
        ]
    )
