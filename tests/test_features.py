"""Unit tests for FeatureEngineer.
These are the most important tests in the project. FeatureEngineer is the
first step of every pipeline — bugs here affect all four models silently.
"""

import pandas as pd
from sklearn.base import clone

from credit_risk_model.processing.features import FeatureEngineer


def test_columns_are_dropped(raw_sample):
    """Uninformative columns must be removed."""
    fe = FeatureEngineer()
    result = fe.transform(raw_sample.drop(columns=["class"]))
    for col in ["other_debtors_guarantors", "telephone", "people_liable_for_maintenance"]:
        assert col not in result.columns, f"Column '{col}' should have been dropped"


def test_no_checking_binary(raw_sample):
    """no_checking must be 0 or 1 only."""
    fe = FeatureEngineer()
    result = fe.transform(raw_sample.drop(columns=["class"]))
    assert set(result["no_checking"].unique()).issubset({0, 1})


def test_log_transforms_non_negative(raw_sample):
    """Log-transformed features must be >= 0 (since we use log1p)."""
    fe = FeatureEngineer()
    result = fe.transform(raw_sample.drop(columns=["class"]))
    for col in ["credit_log", "duration_log", "monthly_burden_log"]:
        assert (result[col] >= 0).all(), f"{col} contains negative values"


def test_purpose_categories_merged(raw_sample):
    """education/retraining → personal_development;
    appliances/repairs/others → home_improvement."""
    fe = FeatureEngineer()
    result = fe.transform(raw_sample.drop(columns=["class"]))
    forbidden = {"education", "retraining", "appliances", "repairs", "others"}
    remaining = set(result["purpose"].unique())
    assert remaining.isdisjoint(forbidden), (
        f"Found un-merged purpose categories: {remaining & forbidden}"
    )


def test_savings_bins_consolidated(raw_sample):
    """Savings must collapse to 3 values: <500 DM, >=500 DM, or unknown."""
    fe = FeatureEngineer()
    result = fe.transform(raw_sample.drop(columns=["class"]))
    allowed = {"< 500 DM", ">= 500 DM", "unknown/no savings account"}
    remaining = set(result["savings_account_bonds"].dropna().unique())
    # Only consolidated values should remain; unknown may or may not appear.
    assert remaining.issubset(allowed), f"Unexpected savings categories: {remaining - allowed}"


def test_age_group_covers_all_ages(raw_sample):
    """All rows must have a non-null age_group."""
    fe = FeatureEngineer()
    result = fe.transform(raw_sample.drop(columns=["class"]))
    assert result["age_group"].notna().all(), "Some rows have null age_group"
    assert result["age_group"].isin(["Young", "Early_Career", "Prime", "Mature"]).all()


def test_duplicate_checking_flag(raw_sample):
    """duplicate_checking=True must create checking_2 and personal_status_2."""
    X = raw_sample.drop(columns=["class"])
    fe_default = FeatureEngineer(duplicate_checking=False)
    result_default = fe_default.transform(X)
    assert "checking_2" not in result_default.columns

    fe_dup = FeatureEngineer(duplicate_checking=True)
    result_dup = fe_dup.transform(X)
    assert "checking_2" in result_dup.columns
    assert "personal_status_2" in result_dup.columns


def test_duplicate_amount_flag(raw_sample):
    """duplicate_amount=True must create credit_amount_squared."""
    X = raw_sample.drop(columns=["class"])
    fe_default = FeatureEngineer(duplicate_amount=False)
    result_default = fe_default.transform(X)
    assert "credit_amount_squared" not in result_default.columns

    fe_dup = FeatureEngineer(duplicate_amount=True)
    result_dup = fe_dup.transform(X)
    assert "credit_amount_squared" in result_dup.columns
    assert (result_dup["credit_amount_squared"] >= 0).all()


def test_transform_does_not_mutate_input(raw_sample):
    """transform() must not modify the input DataFrame in place."""
    X = raw_sample.drop(columns=["class"]).copy()
    original_columns = set(X.columns)
    fe = FeatureEngineer()
    fe.transform(X)
    assert set(X.columns) == original_columns, "transform() mutated the input DataFrame"


def test_fit_returns_self(raw_sample):
    """fit() must return self for sklearn Pipeline compatibility."""
    fe = FeatureEngineer()
    X = raw_sample.drop(columns=["class"])
    result = fe.fit(X)
    assert result is fe


def test_fit_learns_credit_amount_bin_edges(raw_sample):
    """fit() should persist stable bin edges for inference-time reuse."""
    fe = FeatureEngineer()
    X = raw_sample.drop(columns=["class"])
    fe.fit(X)
    assert hasattr(fe, "credit_amount_bin_edges_")
    assert fe.credit_amount_bin_edges_ is None or len(fe.credit_amount_bin_edges_) >= 2


def test_single_row_transform_does_not_fail(raw_sample):
    """Single-row inference should not fail on quantile-based bins."""
    single_row = raw_sample.drop(columns=["class"]).head(1)
    fe = FeatureEngineer().fit(raw_sample.drop(columns=["class"]))
    result = fe.transform(single_row)
    assert result["credit_amount_bins"].notna().all()


def test_single_row_transform_is_safe_without_fit(raw_sample):
    """Old pickled transformers should keep single-row inference safe."""
    single_row = raw_sample.drop(columns=["class"]).head(1)
    fe = FeatureEngineer()
    result = fe.transform(single_row)
    assert result["credit_amount_bins"].notna().all()


class TestSklearnContract:
    """Verify FeatureEngineer follows the sklearn estimator interface.

    We don't use parametrize_with_checks because FeatureEngineer is a
    domain-specific transformer that requires pandas DataFrames with named
    columns — sklearn's generic checks feed numpy arrays which aren't
    applicable here. Instead we test the contract points explicitly.
    """

    def test_clone_roundtrip(self):
        """clone() must produce an equivalent unfitted estimator."""
        fe = FeatureEngineer(duplicate_checking=True, duplicate_amount=True)
        cloned = clone(fe)
        assert cloned is not fe
        assert cloned.get_params() == fe.get_params()

    def test_get_set_params(self):
        """get_params / set_params must be symmetric."""
        fe = FeatureEngineer()
        params = fe.get_params()
        assert params == {
            "duplicate_checking": False,
            "duplicate_amount": False,
        }
        fe.set_params(duplicate_checking=True)
        assert fe.duplicate_checking is True
        assert fe.get_params()["duplicate_checking"] is True

    def test_repr(self):
        """repr must be parseable (sklearn convention)."""
        fe = FeatureEngineer(duplicate_checking=True)
        r = repr(fe)
        assert "FeatureEngineer" in r
        assert "duplicate_checking=True" in r

    def test_fit_transform_returns_dataframe(self, raw_sample):
        """fit_transform must work and return a DataFrame."""
        fe = FeatureEngineer()
        result = fe.fit_transform(raw_sample.drop(columns=["class"]))
        assert isinstance(result, pd.DataFrame)
