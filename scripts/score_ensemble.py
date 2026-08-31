"""Final scoring of the ensemble on the hold-out test set.

This script only reports. The ensemble weights and decision threshold in
``model_config.yml`` are chosen on the validation set by
``scripts/select_ensemble.py``, and this script reads that configuration and
scores it once against the test set.

Splitting the two steps across two scripts keeps the reported test cost a genuine
held-out number. To change weights or thresholds, re-run the selection step
rather than re-running this script with different values.

Usage
-----
    python scripts/select_ensemble.py --write   # choose the configuration
    python scripts/score_ensemble.py            # then score it, once
"""

import argparse

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from credit_risk_model.config.core import DATA_DIR, config
from credit_risk_model.ensemble import CreditRiskEnsemble, CreditRiskPyfunc
from credit_risk_model.predict import load_pipelines_from_registry
from credit_risk_model.tracking.metrics import trivial_policy_costs
from credit_risk_model.tracking.visualizations import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
)


def print_summary(metrics: dict, ensemble: CreditRiskEnsemble, n_test: int) -> None:
    """Print the test-set result, with the trivial policies as reference points.

    Everything here is also logged to MLflow. It is printed too so that running
    the script tells you the answer without opening the MLflow UI.
    """
    weights = ", ".join(f"{k.upper()}={v:.1f}" for k, v in ensemble.weights.items())

    print(f"\n{'=' * 68}")
    print("ENSEMBLE ON THE HOLD-OUT TEST SET")
    print(f"{'=' * 68}")
    print(f"   Test set  : n={n_test}")
    print(f"   Weights   : {weights}")
    print(f"   Threshold : {ensemble.threshold:.2f}")

    print(f"\n   {'-' * 44}")
    print("   Reference points (no model)")
    print(f"   {'-' * 44}")
    print(f"   {'Approve everyone':>28s}:  {metrics['always_approve_cost']:>6.0f}")
    print(f"   {'Reject everyone':>28s}:  {metrics['always_reject_cost']:>6.0f}")

    print(f"\n   {'-' * 44}")
    print("   Ensemble")
    print(f"   {'-' * 44}")
    for label, key in (
        ("Total cost", "ensemble_cost"),
        ("ROC AUC", "ensemble_roc_auc"),
        ("Precision", "ensemble_precision"),
        ("Recall", "ensemble_recall"),
        ("Accuracy", "ensemble_accuracy"),
        ("F1", "ensemble_f1"),
    ):
        value = metrics[key]
        formatted = f"{value:>6.0f}" if key == "ensemble_cost" else f"{value:>6.3f}"
        print(f"   {label:>28s}:  {formatted}")
    print()


def score_ensemble(weights: dict | None = None, threshold: float | None = None):
    # Load the four trained pipelines from MLflow registry
    pipelines = load_pipelines_from_registry()

    # Build ensemble with config defaults (or CLI overrides)
    ensemble = CreditRiskEnsemble(
        pipelines=pipelines,
        weights=weights,
        threshold=threshold,
    )

    # Load test data
    test_df = pd.read_csv(DATA_DIR / config.test_data_file)
    X_test = test_df.drop(columns=[config.target])
    y_test = test_df[config.target]

    # Evaluate and log to MLflow
    mlflow.set_tracking_uri(config.mlflow["backend_store_uri"])
    mlflow.set_experiment(config.mlflow["experiment_name"])

    with mlflow.start_run(run_name=f"Ensemble_t{ensemble.threshold:.2f}"):
        mlflow.log_params(
            {
                "threshold": ensemble.threshold,
                "voting_type": "soft",
                **{f"weight_{k}": v for k, v in ensemble.weights.items()},
            }
        )

        metrics = ensemble.evaluate(X_test, y_test)

        # Trivial no-model policies, logged so the ensemble cost has a reference
        # point in MLflow rather than standing alone.
        metrics.update(
            trivial_policy_costs(
                y_test.values,
                cost_fp=config.cost_matrix.false_positive,
                cost_fn=config.cost_matrix.false_negative,
            )
        )
        mlflow.log_metrics(metrics)
        print_summary(metrics, ensemble, len(y_test))

        fig_cm = plot_confusion_matrix(
            y_test.values,
            ensemble.predict(X_test),
            title="Ensemble Confusion Matrix",
        )
        mlflow.log_figure(fig_cm, "confusion_matrix.png")

        fig_pr = plot_precision_recall_curve(
            y_test.values,
            ensemble.predict_proba(X_test),
            title="Ensemble PR Curve",
        )
        mlflow.log_figure(fig_pr, "precision_recall_curve.png")

        # Build an explicit model signature so MLflow doesn't warn
        input_example = X_test.head(1)
        example_output = pd.DataFrame(
            {
                "probability": ensemble.predict_proba(input_example),
                "decision": ensemble.predict(input_example),
                "threshold": ensemble.threshold,
            }
        )
        signature = infer_signature(input_example, example_output)

        # Register the ensemble as a pyfunc model
        mlflow.pyfunc.log_model(
            artifact_path="ensemble_model",
            python_model=CreditRiskPyfunc(ensemble=ensemble),
            registered_model_name="credit-risk-ensemble",
            signature=signature,
            input_example=input_example,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the credit risk ensemble on the test set and log to MLflow.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Diagnostic override of the decision threshold. The shipped value comes "
            "from validation-set tuning, so this is not a way to search for a better "
            "test cost."
        ),
    )
    args = parser.parse_args()
    score_ensemble(threshold=args.threshold)
