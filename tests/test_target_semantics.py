import pytest

from credit_risk_model.target_semantics import (
    BAD_CLASS,
    GOOD_CLASS,
    is_low_risk,
    label_text,
    probability_of_bad,
    risk_status_from_class,
    risk_status_from_probability,
)


def test_probability_of_bad_is_inverse_of_good_probability():
    assert probability_of_bad(0.82) == pytest.approx(0.18)


def test_risk_status_from_class_matches_label_encoding():
    assert risk_status_from_class(GOOD_CLASS) == "Low Risk"
    assert risk_status_from_class(BAD_CLASS) == "High Risk"


def test_risk_status_from_probability_uses_approval_threshold():
    assert is_low_risk(0.84, 0.83) is True
    assert is_low_risk(0.82, 0.83) is False
    assert risk_status_from_probability(0.84, 0.83) == "Low Risk"
    assert risk_status_from_probability(0.82, 0.83) == "High Risk"


def test_label_text_matches_encoded_target_values():
    assert label_text(GOOD_CLASS) == "Good Risk (1)"
    assert label_text(BAD_CLASS) == "Bad Risk (0)"
