from __future__ import annotations

import logging
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from credit_risk_model.config.core import config
from credit_risk_model.ensemble import CreditRiskEnsemble

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────


def load_pipelines_from_registry(
    versions: dict[str, str] | None = None,
) -> dict:
    """Load the four trained pipelines from the MLflow model registry.

    Parameters
    ----------
    versions : dict mapping model key → version string, e.g. {"lrc": "latest"}
        Defaults to "latest" for all models.

    Returns
    -------
    dict mapping model key → fitted sklearn Pipeline
    """
    versions = versions or {}
    mlflow.set_tracking_uri(config.mlflow["backend_store_uri"])

    pipelines = {}
    for model_key, model_cfg in config.models.items():
        version = versions.get(model_key, "latest")
        uri = f"models:/{model_cfg.registry_name}/{version}"
        try:
            pipelines[model_key] = mlflow.sklearn.load_model(uri)
            logger.info(f"Loaded {model_key} from registry: {uri}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{model_key}' from registry '{uri}'. "
                f"Run the training script first. Error: {exc}"
            ) from exc

    return pipelines


def load_pipelines_from_dir(models_dir: Path) -> dict:
    """Load four pickled pipelines from a directory.

    Used by the Streamlit app and any environment without an MLflow server.
    Expects files named: lrc_pipeline.pkl, rfc_pipeline.pkl,
    svc_pipeline.pkl, cat_pipeline.pkl
    """
    pipelines = {}
    for model_key in config.models:
        pkl_path = models_dir / f"{model_key}_pipeline.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Pipeline file not found: {pkl_path}. "
                f"Run scripts/export_pipelines.py to generate pickle files."
            )
        pipelines[model_key] = joblib.load(pkl_path)
        logger.info(f"Loaded {model_key} from {pkl_path}")

    return pipelines


def make_prediction(
    input_data: dict | pd.DataFrame,
    pipelines: dict | None = None,
    models_dir: Path | None = None,
) -> dict:
    """Generate credit risk predictions.

    This is the single public prediction interface. It accepts raw applicant
    data and returns predictions, probabilities, and the operating threshold.

    Parameters
    ----------
    input_data
        Raw feature data as a dict (single applicant) or DataFrame (batch).
    pipelines
        Pre-loaded pipeline dict (fastest path — avoids reloading on each call).
        If None, loads from MLflow registry.
    models_dir
        If provided, loads from pickle files in this directory instead of MLflow.

    Returns
    -------
    dict with keys: predictions, probabilities, threshold, errors
    """
    data = pd.DataFrame([input_data] if isinstance(input_data, dict) else input_data)

    try:
        if pipelines is None:
            if models_dir is not None:
                pipelines = load_pipelines_from_dir(models_dir)
            else:
                pipelines = load_pipelines_from_registry()

        ensemble = CreditRiskEnsemble(pipelines=pipelines)
        breakdown = ensemble.predict_with_breakdown(data)

        return {
            "predictions": breakdown["ensemble_pred"].tolist(),
            "probabilities": breakdown["ensemble_proba"].tolist(),
            "threshold": breakdown["threshold"],
            "model_breakdown": {
                k: v.tolist()
                for k, v in breakdown.items()
                if k.endswith("_proba") and not k.startswith("ensemble")
            },
            "errors": None,
        }

    except Exception as exc:
        logger.error(f"Prediction failed: {exc}")
        return {
            "predictions": None,
            "probabilities": None,
            "threshold": None,
            "errors": str(exc),
        }
