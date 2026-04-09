from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator, model_validator

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PACKAGE_ROOT / "config" / "model_config.yml"
# Resolve data dir relative to project root (two levels above src/)
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


class CostMatrix(BaseModel):
    false_positive: float
    false_negative: float

    @property
    def ratio(self) -> float:
        """FP/FN cost ratio — informative starting point for threshold search."""
        return self.false_positive / self.false_negative


class EnsembleConfig(BaseModel):
    threshold: float
    weights: dict[str, float]  # {"lrc": 2.5, "rfc": 1.5, "svc": 3.0, "cat": 2.0}

    @field_validator("threshold")
    @classmethod
    def threshold_in_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("Threshold must be between 0 and 1")
        return v

    def weights_as_list(self, order: list[str] | None = None) -> list[float]:
        """Return weights in a specified order, defaulting to [lrc, rfc, svc, cat]."""
        order = order or ["lrc", "rfc", "svc", "cat"]
        return [self.weights[k] for k in order]


class SingleModelConfig(BaseModel):
    experiment_name: str
    registry_name: str
    cv_folds: int
    bayes_n_iter: int
    bayes_n_points: int = 5
    use_smote: bool = False
    smote_type: str = "smote"
    duplicate_checking: bool = False
    duplicate_amount: bool = False
    class_weight: dict[int, int] | None = None
    scale_pos_weight: float | None = None
    one_hot_cols: list[str] = []
    woe_cols: list[str] = []
    count_cols: list[str] = []
    target_cols: list[str] = []
    ordinal_cols: list[str] = []
    numeric_scaled_cols: list[str] = []
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    passthrough_cols: list[str] = []


class AppConfig(BaseModel):
    training_data_file: str
    test_data_file: str
    target: str
    random_state: int
    val_size: int
    cost_matrix: CostMatrix
    ensemble: EnsembleConfig
    models: dict[str, SingleModelConfig]
    mlflow: dict[str, str]

    @model_validator(mode="after")
    def ensemble_keys_match_model_keys(self) -> AppConfig:
        """Catch the app.py bug: different number of weights and models at config load time."""
        ensemble_keys = set(self.ensemble.weights.keys())
        config_keys = set(self.models.keys())
        if ensemble_keys != config_keys:
            raise ValueError(
                f"Ensemble weight keys {ensemble_keys} "
                f"don't match model config keys {config_keys}. "
                f"Add missing weights or remove unused model configs."
            )
        return self


def load_config(config_path: Path = CONFIG_FILE) -> AppConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Resolve relative SQLite URI to absolute path so it works
    # from any working directory (main.py, notebooks/, scripts/).
    mlflow_uri = raw.get("mlflow", {}).get("backend_store_uri", "")
    if mlflow_uri.startswith("sqlite:///") and not mlflow_uri.startswith("sqlite:////"):
        db_filename = mlflow_uri.replace("sqlite:///", "")
        abs_db_path = PROJECT_ROOT / db_filename
        raw["mlflow"]["backend_store_uri"] = f"sqlite:///{abs_db_path}"

    return AppConfig(**raw)


config = load_config()
