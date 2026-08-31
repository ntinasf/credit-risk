"""Tests for the cost function that defines the project's objective.

The direction of the cost matrix is the one thing in this repo that is easy to
invert silently and expensive to get wrong: with 1 = good and 0 = bad, the
5x-expensive error is approving a bad borrower (pred=1, true=0). These tests
pin that direction down in both the metric and the shipped config.
"""

import numpy as np
import pandas as pd

from credit_risk_model.config.core import config
from credit_risk_model.target_semantics import BAD_CLASS, GOOD_CLASS
from credit_risk_model.tracking.metrics import (
    calculate_cost,
    evaluate_model,
    trivial_policy_costs,
)


class _FixedProbaPipeline:
    """Minimal stand-in exposing the only method evaluate_model needs."""

    def __init__(self, proba):
        self._proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1.0 - self._proba, self._proba])


# ── Cost direction ────────────────────────────────────────────────────────────


def test_approving_a_bad_borrower_costs_the_false_positive_price():
    total, _ = calculate_cost([BAD_CLASS], [GOOD_CLASS], cost_fp=5.0, cost_fn=1.0)
    assert total == 5.0


def test_rejecting_a_good_borrower_costs_the_false_negative_price():
    total, _ = calculate_cost([GOOD_CLASS], [BAD_CLASS], cost_fp=5.0, cost_fn=1.0)
    assert total == 1.0


def test_correct_predictions_cost_nothing():
    y_true = [GOOD_CLASS, BAD_CLASS, GOOD_CLASS, BAD_CLASS]
    total, avg = calculate_cost(y_true, y_true, cost_fp=5.0, cost_fn=1.0)
    assert total == 0.0
    assert avg == 0.0


def test_approving_every_bad_costs_five_times_rejecting_every_good():
    n = 20
    approve_all_bad, _ = calculate_cost([BAD_CLASS] * n, [GOOD_CLASS] * n)
    reject_all_good, _ = calculate_cost([GOOD_CLASS] * n, [BAD_CLASS] * n)
    assert approve_all_bad == 5 * reject_all_good


def test_average_cost_is_total_divided_by_sample_count():
    y_true = [BAD_CLASS, BAD_CLASS, GOOD_CLASS, GOOD_CLASS]
    y_pred = [GOOD_CLASS, GOOD_CLASS, GOOD_CLASS, GOOD_CLASS]
    total, avg = calculate_cost(y_true, y_pred, cost_fp=5.0, cost_fn=1.0)
    assert total == 10.0
    assert avg == total / len(y_true)


def test_cost_arguments_override_the_defaults():
    total, _ = calculate_cost([BAD_CLASS], [GOOD_CLASS], cost_fp=50.0, cost_fn=1.0)
    assert total == 50.0


# ── Trivial no-model policies ─────────────────────────────────────────────────


def test_trivial_policy_costs_follow_the_class_counts():
    y_true = [GOOD_CLASS] * 10 + [BAD_CLASS] * 4
    costs = trivial_policy_costs(y_true, cost_fp=5.0, cost_fn=1.0)
    assert costs["always_approve_cost"] == 4 * 5.0
    assert costs["always_reject_cost"] == 10 * 1.0


def test_rejecting_everyone_beats_approving_everyone_on_this_dataset():
    """Sanity check on the reference point the README quotes."""
    y_true = [GOOD_CLASS] * 106 + [BAD_CLASS] * 53
    costs = trivial_policy_costs(y_true, cost_fp=5.0, cost_fn=1.0)
    assert costs["always_reject_cost"] < costs["always_approve_cost"]


# ── evaluate_model ────────────────────────────────────────────────────────────


def test_evaluate_model_respects_the_threshold():
    proba = [0.9, 0.8, 0.3, 0.2]
    y = pd.Series([GOOD_CLASS, GOOD_CLASS, BAD_CLASS, BAD_CLASS])
    pipeline = _FixedProbaPipeline(proba)
    X = pd.DataFrame({"unused": range(len(proba))})

    lenient = evaluate_model(pipeline, X, y, threshold=0.5)
    strict = evaluate_model(pipeline, X, y, threshold=0.95)

    assert lenient["cost"] == 0.0
    assert strict["recall"] == 0.0
    assert strict["cost"] == 2.0  # both good applicants rejected, 1 each


# ── Config guards: cost belongs at the threshold, not in the class weights ────


def test_random_forest_weighting_does_not_favour_the_majority_good_class():
    """The original defect: class_weight={0: 1, 1: 5} upweighted good, the majority."""
    class_weight = config.models["rfc"].class_weight
    if isinstance(class_weight, str):
        assert class_weight in {"balanced", "balanced_subsample"}
    else:
        assert class_weight[BAD_CLASS] >= class_weight[GOOD_CLASS], (
            "weighting the good majority class up is the inversion this guards against"
        )


def test_catboost_does_not_upweight_the_majority_good_class():
    scale_pos_weight = config.models["cat"].scale_pos_weight
    assert scale_pos_weight is not None
    assert scale_pos_weight <= 1.0, (
        "CatBoost's positive class is 1 = good and is the majority, so its weight must not exceed 1"
    )


def test_cost_matrix_makes_approving_a_bad_borrower_the_expensive_error():
    assert config.cost_matrix.false_positive > config.cost_matrix.false_negative


def test_ensemble_threshold_is_where_the_cost_asymmetry_shows_up():
    """A 5:1 penalty on approving bad borrowers should push the bar above 0.5."""
    assert config.ensemble.threshold > 0.5
