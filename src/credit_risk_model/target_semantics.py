from __future__ import annotations

GOOD_CLASS = 1
BAD_CLASS = 0


def probability_of_bad(good_probability: float) -> float:
    """Convert P(good credit) into P(default / bad credit)."""
    return 1.0 - float(good_probability)


def is_low_risk(good_probability: float, threshold: float) -> bool:
    """Return True when approval probability clears the threshold."""
    return float(good_probability) >= float(threshold)


def risk_status_from_probability(
    good_probability: float,
    threshold: float,
) -> str:
    """Map approval probability to the app-facing risk label."""
    return "Low Risk" if is_low_risk(good_probability, threshold) else "High Risk"


def risk_status_from_class(predicted_class: int) -> str:
    """Map encoded class labels to the app-facing risk label."""
    return "Low Risk" if int(predicted_class) == GOOD_CLASS else "High Risk"


def label_text(target_class: int) -> str:
    """Return a readable label for the encoded target value."""
    return (
        f"Good Risk ({GOOD_CLASS})"
        if int(target_class) == GOOD_CLASS
        else f"Bad Risk ({BAD_CLASS})"
    )
