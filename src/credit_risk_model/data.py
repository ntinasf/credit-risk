"""Canonical data splits.

Every stage of the pipeline has to see the same train/validation split, otherwise
the "selected on validation, scored once on test" guarantee quietly breaks.
main.py, the selection script and the notebook all call in here instead of
repeating `train_test_split(...)` three times.

The split is deterministic given `random_state` and `val_size` in
model_config.yml, so re-running any stage reproduces the same partition.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk_model.config.core import DATA_DIR, AppConfig, config


def load_training_frame(app_config: AppConfig = config) -> pd.DataFrame:
    """Read train_data.csv (the 841 rows reserved for fitting and validation)."""
    path = DATA_DIR / app_config.training_data_file
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. Run:\n"
            "  python scripts/process_data.py\n"
            "  python scripts/split_data.py"
        )
    return pd.read_csv(path)


def load_train_validation_split(
    app_config: AppConfig = config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return (X_train, X_val, y_train, y_val), stratified and deterministic.

    The validation set is never used for fitting. It carries every selection
    decision: per-model decision thresholds, ensemble weights and the ensemble
    threshold.
    """
    df = load_training_frame(app_config)
    X = df.drop(columns=[app_config.target])
    y = df[app_config.target]

    return train_test_split(
        X,
        y,
        test_size=app_config.val_size,
        random_state=app_config.random_state,
        stratify=y,
    )


def load_test_split(
    app_config: AppConfig = config,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X_test, y_test) — the hold-out set, scored once at the very end."""
    path = DATA_DIR / app_config.test_data_file
    if not path.exists():
        raise FileNotFoundError(f"Test data not found at {path}. Run scripts/split_data.py first.")
    df = pd.read_csv(path)
    return df.drop(columns=[app_config.target]), df[app_config.target]
