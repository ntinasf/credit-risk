"""Export trained pipelines from MLflow registry to pickle files.

This script downloads the four trained pipelines from the MLflow model
registry and saves them as .pkl files in a target directory (default:
``app/models/``).  The Streamlit app and any other non-MLflow consumer
loads models from these pickle files via ``load_pipelines_from_dir()``.

Usage
-----
    # Export latest versions to app/models/
    python scripts/export_pipelines.py

    # Export to a custom directory
    python scripts/export_pipelines.py --output-dir /path/to/models

    # Export specific model versions
    python scripts/export_pipelines.py --versions lrc=3 rfc=2 svc=1 cat=latest
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib

from credit_risk_model.config.core import config
from credit_risk_model.predict import load_pipelines_from_registry

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "app" / "models"


def export_pipelines(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    versions: dict[str, str] | None = None,
) -> None:
    """Load pipelines from MLflow and save as pickle files.

    Parameters
    ----------
    output_dir
        Directory to write ``<key>_pipeline.pkl`` files.
    versions
        Mapping of model key → version string.  ``None`` = "latest" for all.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading pipelines from MLflow registry …")
    pipelines = load_pipelines_from_registry(versions=versions)

    for model_key, pipeline in pipelines.items():
        pkl_path = output_dir / f"{model_key}_pipeline.pkl"
        joblib.dump(pipeline, pkl_path)
        logger.info(f"  ✓ {model_key} → {pkl_path}")

    logger.info(f"Done — {len(pipelines)} pipelines exported to {output_dir}")


def _parse_versions(version_args: list[str] | None) -> dict[str, str] | None:
    """Parse ``['lrc=3', 'rfc=latest']`` into ``{'lrc': '3', 'rfc': 'latest'}``."""
    if not version_args:
        return None
    versions: dict[str, str] = {}
    for item in version_args:
        key, _, ver = item.partition("=")
        if key not in config.models:
            raise ValueError(
                f"Unknown model key '{key}'. "
                f"Valid keys: {list(config.models.keys())}"
            )
        versions[key] = ver or "latest"
    return versions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export trained pipelines from MLflow registry to pickle files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for .pkl files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--versions",
        nargs="*",
        metavar="KEY=VERSION",
        help="Model versions, e.g. lrc=3 rfc=latest  (default: latest for all)",
    )
    args = parser.parse_args()
    export_pipelines(
        output_dir=args.output_dir,
        versions=_parse_versions(args.versions),
    )
