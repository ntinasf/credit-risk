"""Unit tests for CreditRiskEnsemble combination and threshold logic."""

import numpy as np
import pytest

from credit_risk_model.ensemble import CreditRiskEnsemble


def test_ensemble_probabilities_in_range(mock_pipelines, X_train_small):
    """Weighted average probabilities must be in [0, 1]."""
    ensemble = CreditRiskEnsemble(pipelines=mock_pipelines)
    proba = ensemble.predict_proba(X_train_small)
    assert np.all(proba >= 0.0), "Probabilities below 0"
    assert np.all(proba <= 1.0), "Probabilities above 1"


def test_ensemble_predict_returns_binary(mock_pipelines, X_train_small):
    """predict() must return only 0s and 1s."""
    ensemble = CreditRiskEnsemble(pipelines=mock_pipelines)
    preds = ensemble.predict(X_train_small)
    assert set(np.unique(preds)).issubset({0, 1})


def test_ensemble_rejects_mismatched_weights(mock_pipelines):
    """Constructor must raise if weight keys don't match pipeline keys."""
    bad_weights = {"lrc": 2.5, "rfc": 1.5, "svc": 3.0}  # missing 'cat'
    with pytest.raises(ValueError, match="don't match weight keys"):
        CreditRiskEnsemble(pipelines=mock_pipelines, weights=bad_weights)


def test_higher_weight_model_influences_output(mock_pipelines, X_train_small):
    """Changing weights must change ensemble probabilities."""
    ensemble_a = CreditRiskEnsemble(
        pipelines=mock_pipelines,
        weights={"lrc": 10.0, "rfc": 0.0, "svc": 0.0, "cat": 0.0},
    )
    ensemble_b = CreditRiskEnsemble(
        pipelines=mock_pipelines,
        weights={"lrc": 0.0, "rfc": 0.0, "svc": 0.0, "cat": 10.0},
    )
    proba_a = ensemble_a.predict_proba(X_train_small)
    proba_b = ensemble_b.predict_proba(X_train_small)
    # At least some predictions must differ
    assert not np.allclose(proba_a, proba_b, atol=1e-6)


def test_threshold_affects_predictions(mock_pipelines, X_train_small):
    """A lower threshold must produce more positive predictions (or equal)."""
    ensemble_low = CreditRiskEnsemble(pipelines=mock_pipelines, threshold=0.2)
    ensemble_high = CreditRiskEnsemble(pipelines=mock_pipelines, threshold=0.9)
    n_positive_low = ensemble_low.predict(X_train_small).sum()
    n_positive_high = ensemble_high.predict(X_train_small).sum()
    assert n_positive_low >= n_positive_high


def test_optimize_threshold_changes_default(mock_pipelines, X_train_small, y_train_small):
    """optimize_threshold() should update self.threshold from the default."""
    ensemble = CreditRiskEnsemble(pipelines=mock_pipelines, threshold=0.5)
    new_threshold = ensemble.optimize_threshold(X_train_small, y_train_small)
    # The threshold was updated in place
    assert ensemble.threshold == new_threshold
    # On cost-sensitive data (5:1 ratio), threshold typically shifts from 0.5
    assert 0.0 < new_threshold < 1.0
