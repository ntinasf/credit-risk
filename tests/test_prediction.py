"""Integration tests for the end-to-end prediction interface."""

import pandas as pd

from credit_risk_model.predict import make_prediction


def test_make_prediction_returns_correct_structure(mock_pipelines, X_train_small):
    """make_prediction() must return a dict with all required keys."""
    result = make_prediction(
        input_data=X_train_small.head(5),
        pipelines=mock_pipelines,
    )
    assert result["errors"] is None
    assert result["predictions"] is not None
    assert result["probabilities"] is not None
    assert len(result["predictions"]) == 5
    assert len(result["probabilities"]) == 5
    assert 0.0 < result["threshold"] < 1.0


def test_predictions_are_binary(mock_pipelines, X_train_small):
    """Predictions must be 0 or 1."""
    result = make_prediction(input_data=X_train_small.head(20), pipelines=mock_pipelines)
    assert all(p in (0, 1) for p in result["predictions"])


def test_probabilities_in_range(mock_pipelines, X_train_small):
    """Probabilities must be in [0, 1]."""
    result = make_prediction(input_data=X_train_small.head(20), pipelines=mock_pipelines)
    for p in result["probabilities"]:
        assert 0.0 <= p <= 1.0, f"Probability {p} out of range"


def test_model_breakdown_contains_all_four_models(mock_pipelines, X_train_small):
    """The breakdown dict must contain probabilities for all four models."""
    result = make_prediction(input_data=X_train_small.head(5), pipelines=mock_pipelines)
    breakdown = result["model_breakdown"]
    for model_key in ["lrc", "rfc", "svc", "cat"]:
        assert f"{model_key}_proba" in breakdown, f"Missing breakdown for {model_key}"


def test_make_prediction_handles_errors_gracefully(mock_pipelines):
    """A broken input must return errors without raising an exception."""
    bad_input = pd.DataFrame({"nonexistent_column": [1, 2, 3]})
    result = make_prediction(input_data=bad_input, pipelines=mock_pipelines)
    assert result["errors"] is not None
    assert result["predictions"] is None
