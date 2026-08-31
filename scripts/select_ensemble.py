"""Select the ensemble weights and decision threshold on the VALIDATION set.

This is the authoritative selection step. It is the only place the shipped
ensemble configuration is decided, and it never touches the test set.

Procedure (mirrors notebooks/investigation.ipynb section 8):

1. Load the four registered pipelines from MLflow. These are the same artifacts
   that get exported and served, so the configuration is selected for the models
   that actually ship.
2. Sweep every weight combination on the validation set and shortlist the top N
   by ROC AUC, which is threshold-independent because the threshold is still to
   be chosen.
3. Tune each shortlisted candidate's decision threshold on the validation set to
   minimise expected cost.
4. Promote the lowest-cost candidate (precision, then ROC AUC break ties).
5. With ``--write``, write the promoted weights and threshold into
   ``model_config.yml``, which is what the scoring script and the app read.

The run is deterministic: the validation split comes from
``load_train_validation_split()`` and the weight grid is fixed, so re-running
reproduces the same configuration.

Usage
-----
    python scripts/select_ensemble.py            # report only
    python scripts/select_ensemble.py --write    # update model_config.yml
"""

from __future__ import annotations

import argparse
import logging
import re
from itertools import product
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from credit_risk_model.config.core import CONFIG_FILE, config
from credit_risk_model.data import load_train_validation_split
from credit_risk_model.ensemble import CreditRiskEnsemble
from credit_risk_model.predict import load_pipelines_from_registry
from credit_risk_model.tracking.metrics import calculate_cost

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

MODEL_ORDER = ["lrc", "rfc", "svc", "cat"]
WEIGHT_GRID = [1.0, 1.5, 2.0, 2.5]
SHORTLIST_SIZE = 15
SHORTLIST_THRESHOLD = 0.5  # only used for the tie-break diagnostics


def sweep_weights(probas: dict[str, np.ndarray], y_val: pd.Series) -> pd.DataFrame:
    """Score every weight combination on the validation set.

    Ranked by ROC AUC. Cost does not come in here; it belongs to the threshold,
    which has not been chosen yet.
    """
    records = []
    for weights in product(WEIGHT_GRID, repeat=len(MODEL_ORDER)):
        total = sum(weights)
        blended = sum(w * probas[k] for w, k in zip(weights, MODEL_ORDER, strict=True)) / total
        at_default = (blended >= SHORTLIST_THRESHOLD).astype(int)

        records.append(
            {
                **{f"w_{k}": w for k, w in zip(MODEL_ORDER, weights, strict=True)},
                "roc_auc": roc_auc_score(y_val, blended),
                "average_precision": average_precision_score(y_val, blended),
                "precision_default": precision_score(y_val, at_default, zero_division=0),
                "f1_default": f1_score(y_val, at_default, zero_division=0),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            ["roc_auc", "average_precision", "precision_default", "f1_default"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def tune_candidates(
    shortlist: pd.DataFrame,
    pipelines: dict,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> pd.DataFrame:
    """Tune each candidate's threshold on validation, then rank by cost.

    Threshold tuning reuses ``CreditRiskEnsemble.optimize_threshold`` so the
    selection logic and the serving class cannot drift apart.
    """
    cost_fp = config.cost_matrix.false_positive
    cost_fn = config.cost_matrix.false_negative

    rows = []
    for rank, row in shortlist.iterrows():
        weights = {k: float(row[f"w_{k}"]) for k in MODEL_ORDER}
        ensemble = CreditRiskEnsemble(pipelines=pipelines, weights=weights, threshold=0.5)
        threshold = ensemble.optimize_threshold(X_val, y_val)

        proba = ensemble.predict_proba(X_val)
        preds = (proba >= threshold).astype(int)
        total_cost, avg_cost = calculate_cost(y_val.values, preds, cost_fp, cost_fn)

        rows.append(
            {
                "candidate": f"C{rank + 1}",
                **{f"w_{k}": weights[k] for k in MODEL_ORDER},
                "threshold": threshold,
                "cost": total_cost,
                "avg_cost": avg_cost,
                "roc_auc": roc_auc_score(y_val, proba),
                "precision": precision_score(y_val, preds, zero_division=0),
                "recall": recall_score(y_val, preds, zero_division=0),
            }
        )

    # Cost is the objective; precision and ROC AUC only break ties.
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["cost", "precision", "roc_auc"],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )


def write_config(weights: dict[str, float], threshold: float, path: Path = CONFIG_FILE) -> None:
    """Rewrite the `ensemble:` block in model_config.yml, preserving comments.

    Targeted line replacement rather than a YAML round-trip, because dumping the
    parsed document would strip every comment in the file.
    """
    missing = set(MODEL_ORDER) - set(weights)
    if missing:
        raise ValueError(f"weights is missing entries for {sorted(missing)}")

    text = path.read_text()
    weight_lines = "\n".join(f"    {k}: {weights[k]}" for k in MODEL_ORDER)
    replacement = f"ensemble:\n  threshold: {threshold}\n  weights:\n{weight_lines}\n"

    # Consume the header plus its indented, non-blank body only, leaving any
    # blank-line separator before the next top-level key untouched.
    pattern = re.compile(
        r"^ensemble:\n(?:[ \t]+\S.*\n)*",
        re.MULTILINE,
    )
    updated, n = pattern.subn(replacement, text, count=1)
    if n != 1:
        raise RuntimeError(f"Could not locate the 'ensemble:' block in {path}")

    path.write_text(updated)
    logger.info(f"Wrote weights and threshold to {path}")


def select_ensemble(write: bool = False, shortlist_size: int = SHORTLIST_SIZE) -> dict:
    X_train, X_val, y_train, y_val = load_train_validation_split()
    logger.info(
        f"Validation set: n={len(y_val)} "
        f"({int((y_val == 1).sum())} good / {int((y_val == 0).sum())} bad)"
    )

    pipelines = load_pipelines_from_registry()
    probas = {k: pipelines[k].predict_proba(X_val)[:, 1] for k in MODEL_ORDER}

    print("\nIndividual models on validation:")
    for k in MODEL_ORDER:
        print(f"   {k.upper():>4s}: ROC AUC = {roc_auc_score(y_val, probas[k]):.4f}")

    sweep = sweep_weights(probas, y_val)
    shortlist = sweep.head(shortlist_size)
    logger.info(f"Swept {len(sweep)} weight combinations, shortlisted {len(shortlist)} by ROC AUC")

    tuned = tune_candidates(shortlist, pipelines, X_val, y_val)
    winner = tuned.iloc[0]

    print("\n" + "=" * 88)
    print(f"PROMOTION — {len(tuned)} candidates, threshold tuned on validation")
    print("=" * 88)
    print(
        tuned[
            [
                "candidate",
                *[f"w_{k}" for k in MODEL_ORDER],
                "threshold",
                "cost",
                "roc_auc",
                "precision",
                "recall",
            ]
        ]
        .head(6)
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )

    weights = {k: float(winner[f"w_{k}"]) for k in MODEL_ORDER}
    threshold = round(float(winner["threshold"]), 2)

    print(f"\nPROMOTED: {winner['candidate']}")
    print(f"   weights   : {weights}")
    print(f"   threshold : {threshold:.2f}")
    print(f"   val cost  : {winner['cost']:.0f}   val ROC AUC: {winner['roc_auc']:.4f}")

    mlflow.set_tracking_uri(config.mlflow["backend_store_uri"])
    mlflow.set_experiment(config.mlflow["experiment_name"])
    with mlflow.start_run(run_name=f"Selection_val_t{threshold:.2f}"):
        mlflow.log_params(
            {
                "stage": "selection",
                "selected_on": "validation",
                "n_val": len(y_val),
                "n_combinations": len(sweep),
                "shortlist_size": len(shortlist),
                "shortlist_metric": "roc_auc",
                "promotion_metric": "cost",
                "threshold": threshold,
                **{f"weight_{k}": v for k, v in weights.items()},
            }
        )
        mlflow.log_metrics(
            {
                "val_cost": float(winner["cost"]),
                "val_roc_auc": float(winner["roc_auc"]),
                "val_precision": float(winner["precision"]),
                "val_recall": float(winner["recall"]),
            }
        )

    if write:
        write_config(weights, threshold)
    else:
        print("\n(dry run, pass --write to update model_config.yml)")

    return {"weights": weights, "threshold": threshold, "val_cost": float(winner["cost"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select ensemble weights and threshold on the validation set.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the promoted configuration into model_config.yml",
    )
    parser.add_argument(
        "--shortlist-size",
        type=int,
        default=SHORTLIST_SIZE,
        help=f"Candidates to carry into threshold tuning (default: {SHORTLIST_SIZE})",
    )
    args = parser.parse_args()
    select_ensemble(write=args.write, shortlist_size=args.shortlist_size)
