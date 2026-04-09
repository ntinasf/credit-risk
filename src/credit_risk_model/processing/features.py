from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Domain-specific feature engineering for the German Credit Dataset.
    Applied as the first step in every model pipeline, before encoding.
    Creates derived features, merges sparse categories, and bins continuous
    variables based on domain knowledge and EDA findings.

    Parameters
    ----------
    duplicate_checking : bool, default False
        Creates `checking_2` and `personal_status_2` duplicate columns.
        RFC uses these for ordinal encoding alongside the original one-hot.
    duplicate_amount : bool, default False
        Creates `credit_amount_squared` polynomial feature.
        SVC uses this non-linear term — it helps the RBF kernel capture
        the non-linear relationship between credit amount and risk.
    """

    def __init__(
        self,
        duplicate_checking: bool = False,
        duplicate_amount: bool = False,
    ) -> None:
        self.duplicate_checking = duplicate_checking
        self.duplicate_amount = duplicate_amount

    def fit(self, X: pd.DataFrame, y=None) -> FeatureEngineer:
        self.credit_amount_bin_edges_ = self._compute_credit_amount_bin_edges(X["credit_amount"])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # ── Drop uninformative features ───────────────────────────────────
        drop_cols = [
            "other_debtors_guarantors",
            "telephone",
            "people_liable_for_maintenance",
        ]
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])

        # ── Binary indicator ──────────────────────────────────────────────
        X["no_checking"] = (X["checking_account_status"] == "no checking account").astype(int)

        # ── Ratio and polynomial features ─────────────────────────────────
        X["monthly_burden"] = X["credit_amount"] / X["duration_months"]
        X["duration_to_age_ratio"] = X["duration_months"] / X["age_years"]
        X["duration_to_age_ratio_sqrt"] = np.sqrt(X["duration_to_age_ratio"])
        X["duration_squared"] = X["duration_months"] ** 2

        if self.duplicate_amount:
            X["credit_amount_squared"] = X["credit_amount"] ** 2

        # ── Log transforms ─────────────────────────────────────────────────
        X["credit_log"] = np.log1p(X["credit_amount"])
        X["duration_log"] = np.log1p(X["duration_months"])
        X["monthly_burden_log"] = np.log1p(X["monthly_burden"])

        # ── Duplicate columns for models that benefit from them ───────────
        if self.duplicate_checking:
            # RFC uses checking_account_status twice: once as one-hot
            # (captures the "no checking account" signal) and once as ordinal
            # (preserves the balance-tier ordering)
            X["checking_2"] = X["checking_account_status"].copy()
            X["personal_status_2"] = X["personal_status_sex"].copy()

        # ── Category consolidation ─────────────────────────────────────────
        # Merge rare purpose categories to reduce sparsity in encoders
        purpose_map = {
            "education": "personal_development",
            "retraining": "personal_development",
            "appliances": "home_improvement",
            "repairs": "home_improvement",
            "others": "home_improvement",
        }
        X["purpose"] = X["purpose"].replace(purpose_map)

        # Savings: 4 levels → 2 (threshold: 500 DM)
        low_savings = {"< 100 DM", "100 - 500 DM"}
        high_savings = {"500 - 1000 DM", ">= 1000 DM"}
        X["savings_account_bonds"] = X["savings_account_bonds"].apply(
            lambda v: "< 500 DM" if v in low_savings else (">= 500 DM" if v in high_savings else v)
        )

        # Housing: 3 levels → 2 (own vs. not own)
        X["housing"] = X["housing"].replace({"for free": "not_own", "rent": "not_own"})

        # Credit history: merge two "all paid" variants for cleaner WOE
        all_paid_variants = {
            "all credits at this bank paid back duly",
            "no credits taken/all credits paid back duly",
        }
        X["credit_history"] = X["credit_history"].apply(
            lambda v: "all credits paid" if v in all_paid_variants else v
        )

        # ── Binning ────────────────────────────────────────────────────────
        X["credit_amount_bins"] = self._transform_credit_amount_bins(X["credit_amount"])
        X["age_group"] = pd.cut(
            X["age_years"],
            bins=[0, 25, 35, 50, 200],
            labels=["Young", "Early_Career", "Prime", "Mature"],
        )
        return X

    @staticmethod
    def _compute_credit_amount_bin_edges(
        series: pd.Series,
    ) -> np.ndarray | None:
        clean = pd.Series(series).dropna().astype(float)
        if clean.empty:
            return None

        quantiles = np.linspace(0, 1, 6)
        bin_edges = np.unique(clean.quantile(quantiles).to_numpy(dtype=float))

        if bin_edges.size < 2:
            return None

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        return bin_edges

    def _transform_credit_amount_bins(
        self,
        series: pd.Series,
    ) -> pd.Categorical:
        labels = ["a", "b", "c", "d", "e"]
        bin_edges = getattr(self, "credit_amount_bin_edges_", None)

        if bin_edges is None:
            bin_edges = self._compute_credit_amount_bin_edges(series)

        if bin_edges is None:
            fallback = pd.Series(["c"] * len(series), index=series.index)
            return pd.Categorical(fallback, categories=labels, ordered=True)

        active_labels = labels[: len(bin_edges) - 1]
        return pd.cut(
            series,
            bins=bin_edges,
            labels=active_labels,
            include_lowest=True,
        )


class BaselineEngineer(TransformerMixin, BaseEstimator):
    """Minimal feature engineer for baseline comparison experiments.
    Applies only the column drops and the no_checking binary flag.
    Use this to isolate the contribution of the full FeatureEngineer
    by comparing model performance with and without it.
    """

    def fit(self, X: pd.DataFrame, y=None) -> BaselineEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        drop_cols = ["other_debtors_guarantors", "telephone", "people_liable_for_maintenance"]
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])
        X["no_checking"] = (X["checking_account_status"] == "no checking account").astype(int)
        return X
