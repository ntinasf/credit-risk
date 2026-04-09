import argparse
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from credit_risk_model.config.core import config, DATA_DIR
from credit_risk_model.ensemble import CreditRiskEnsemble, CreditRiskPyfunc
from credit_risk_model.predict import load_pipelines_from_registry
from credit_risk_model.tracking.visualizations import plot_confusion_matrix, plot_precision_recall_curve


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
        mlflow.log_params({
            "threshold": ensemble.threshold,
            "voting_type": "soft",
            **{f"weight_{k}": v for k, v in ensemble.weights.items()},
        })

        metrics = ensemble.evaluate(X_test, y_test)
        mlflow.log_metrics(metrics)

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
        example_output = pd.DataFrame({
            "probability": ensemble.predict_proba(input_example),
            "decision": ensemble.predict(input_example),
            "threshold": ensemble.threshold,
        })
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
        help="Override ensemble decision threshold (default: from config)",
    )
    args = parser.parse_args()
    score_ensemble(threshold=args.threshold)
