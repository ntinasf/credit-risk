"""Training orchestrator for the Credit Risk Ensemble.

Loads processed data, splits a validation set, trains all four models
via BayesSearchCV, and registers them in MLflow.

Usage
-----
    # Train all models with Bayesian tuning (default)
    uv run python main.py

    # Train without hyperparameter tuning (fast, ~1 min)
    uv run python main.py --no-tune

    # Train a single model
    uv run python main.py --model lrc
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

# Suppress noisy loky worker-timeout warnings (benign on macOS/Apple Silicon)
warnings.filterwarnings(
    "ignore",
    message="A worker stopped while some jobs were given to the executor",
    category=UserWarning,
)

from credit_risk_model.config.core import DATA_DIR, config
from credit_risk_model.training.train_catboost import CatBoostTrainer
from credit_risk_model.training.train_lrc import LRCTrainer
from credit_risk_model.training.train_rf import RFTrainer
from credit_risk_model.training.train_svc import SVCTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TRAINERS = {
    "lrc": LRCTrainer,
    "rfc": RFTrainer,
    "svc": SVCTrainer,
    "cat": CatBoostTrainer,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train credit risk models and register in MLflow.",
    )
    parser.add_argument(
        "--model",
        choices=list(TRAINERS.keys()),
        default=None,
        help="Train a single model (default: all four)",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip Bayesian hyperparameter search (use defaults)",
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────
    train_path = DATA_DIR / config.training_data_file
    if not train_path.exists():
        logger.error(
            f"Training data not found at {train_path}.\n"
            "Run the data pipeline first:\n"
            "  uv run python scripts/process_data.py\n"
            "  uv run python scripts/split_data.py"
        )
        sys.exit(1)

    df = pd.read_csv(train_path)
    X = df.drop(columns=[config.target])
    y = df[config.target]

    logger.info(f"Loaded {len(df)} training samples from {train_path}")

    # ── Hold out a validation set (not used in CV) ────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.val_size,
        random_state=config.random_state,
        stratify=y,
    )
    logger.info(
        f"Split: {len(X_train)} train / {len(X_val)} val  "
        f"(val class balance: {y_val.mean():.1%} positive)"
    )

    # ── Train ─────────────────────────────────────────────────────────────
    models_to_train = (
        {args.model: TRAINERS[args.model]} if args.model else TRAINERS
    )

    for key, TrainerClass in models_to_train.items():
        logger.info(f"{'━' * 60}")
        logger.info(f"Training {key.upper()}…")
        logger.info(f"{'━' * 60}")

        trainer = TrainerClass()
        trainer.train(
            X_train,
            y_train,
            X_val,
            y_val,
            tune=not args.no_tune,
        )

    logger.info(f"\n{'━' * 60}")
    logger.info("✅ All done. Models registered in MLflow.")
    logger.info(
        "Next steps:\n"
        "  uv run python scripts/score_ensemble.py   # evaluate ensemble\n"
        "  uv run python scripts/export_pipelines.py  # export .pkl files\n"
        "  uv run streamlit run app/streamlit_app.py   # launch app"
    )


if __name__ == "__main__":
    main()
