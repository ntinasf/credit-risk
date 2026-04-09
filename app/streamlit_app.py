from pathlib import Path

import pandas as pd
import streamlit as st

from credit_risk_model.config.core import config
from credit_risk_model.predict import load_pipelines_from_dir, make_prediction
from credit_risk_model.target_semantics import (
    BAD_CLASS,
    GOOD_CLASS,
    label_text,
    probability_of_bad,
    risk_status_from_probability,
)

# ============================================================
# PATHS & CONFIG
# ============================================================

MODELS_DIR = Path(__file__).parent / "models"

# ============================================================
# FEATURE DEFINITIONS  (German Credit Dataset)
# ============================================================

CATEGORICAL_FEATURES = {
    "checking_account_status": [
        "no checking account",
        "< 0 DM",
        "0 <= ... < 200 DM",
        ">= 200 DM / salary assign.",
    ],
    "credit_history": [
        "no credits taken/all credits paid back duly",
        "all credits at this bank paid back duly",
        "existing credits paid duly",
        "delay in paying off in past",
        "critical/other credits exist",
    ],
    "purpose": [
        "car (new)",
        "car (used)",
        "furniture/equipment",
        "radio/television",
        "domestic appliances",
        "repairs",
        "education",
        "retraining",
        "business",
        "others",
    ],
    "savings_account_bonds": [
        "unknown/no savings account",
        "< 100 DM",
        "100 - 500 DM",
        "500 - 1000 DM",
        ">= 1000 DM",
    ],
    "present_employment_since": [
        "unemployed",
        "< 1 year",
        "1 <= ... < 4 years",
        "4 <= ... < 7 years",
        ">= 7 years",
    ],
    "personal_status_sex": [
        "male: divorced/separated",
        "female: div/sep/married",
        "male: single",
        "male: married/widowed",
    ],
    "other_debtors_guarantors": [
        "none",
        "co-applicant",
        "guarantor",
    ],
    "property": [
        "real estate",
        "bldg society/life ins.",
        "car or other",
        "unknown/no property",
    ],
    "other_installment_plans": [
        "bank",
        "stores",
        "none",
    ],
    "housing": [
        "rent",
        "own",
        "for free",
    ],
    "job": [
        "unemployed/unskilled non-res.",
        "unskilled resident",
        "skilled employee/official",
        "management/self-employed/etc",
    ],
    "telephone": [
        "none",
        "yes, registered",
    ],
    "foreign_worker": [
        "yes",
        "no",
    ],
}

NUMERIC_FEATURES = {
    "duration_months": {"min": 4, "max": 72, "default": 24, "step": 1},
    "credit_amount": {"min": 250, "max": 20000, "default": 3000, "step": 100},
    "installment_rate_pct_of_disp_income": {
        "min": 1,
        "max": 4,
        "default": 3,
        "step": 1,
    },
    "present_residence_since": {"min": 1, "max": 4, "default": 2, "step": 1},
    "age_years": {"min": 18, "max": 80, "default": 35, "step": 1},
    "existing_credits_count": {"min": 1, "max": 4, "default": 1, "step": 1},
    "people_liable_for_maintenance": {
        "min": 1,
        "max": 2,
        "default": 1,
        "step": 1,
    },
}

FEATURE_LABELS = {
    "checking_account_status": "Checking Account Status",
    "duration_months": "Loan Duration (months)",
    "credit_history": "Credit History",
    "purpose": "Loan Purpose",
    "credit_amount": "Credit Amount (DM)",
    "savings_account_bonds": "Savings Account / Bonds",
    "present_employment_since": "Present Employment Since",
    "installment_rate_pct_of_disp_income": "Installment Rate (% of Income)",
    "present_residence_since": "Present Residence Since (years)",
    "personal_status_sex": "Personal Status & Sex",
    "other_debtors_guarantors": "Other Debtors / Guarantors",
    "property": "Property",
    "age_years": "Age (years)",
    "other_installment_plans": "Other Installment Plans",
    "housing": "Housing",
    "existing_credits_count": "Existing Credits at Bank",
    "job": "Job",
    "people_liable_for_maintenance": "People Liable for Maintenance",
    "telephone": "Telephone",
    "foreign_worker": "Foreign Worker",
}

FEATURE_HELP = {
    "checking_account_status": ("Status of the applicant's checking account at the bank"),
    "duration_months": "Duration of the loan in months",
    "credit_history": "Past credit behavior and repayment history",
    "purpose": "Purpose for which the loan is being requested",
    "credit_amount": "Total amount of credit requested in Deutsche Marks",
    "savings_account_bonds": "Amount in savings accounts or bonds",
    "present_employment_since": "Duration of current employment",
    "installment_rate_pct_of_disp_income": (
        "Loan installment as percentage of disposable income (1-4)"
    ),
    "present_residence_since": "Years at current residence (1-4 scale)",
    "personal_status_sex": "Gender and marital status of applicant",
    "other_debtors_guarantors": ("Whether there are co-applicants or guarantors"),
    "property": "Most valuable property owned by applicant",
    "age_years": "Age of the applicant in years",
    "other_installment_plans": "Other ongoing installment plans (bank/stores)",
    "housing": "Current housing situation of applicant",
    "existing_credits_count": "Number of existing credits at this bank (1-4)",
    "job": "Employment type and skill level",
    "people_liable_for_maintenance": "Number of dependents (1-2)",
    "telephone": "Whether applicant has a registered telephone",
    "foreign_worker": "Whether applicant is a foreign worker",
}

MODEL_LABELS = {
    "lrc": "Logistic Regression",
    "rfc": "Random Forest",
    "svc": "Support Vector Machine",
    "cat": "CatBoost",
}

# ============================================================
# CACHED LOADERS
# ============================================================


@st.cache_resource
def get_pipelines():
    """Load once per Streamlit session. Cached across reruns."""
    return load_pipelines_from_dir(MODELS_DIR)


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(Path(__file__).parent / "data" / "sample_data.csv")


# ============================================================
# UI COMPONENTS
# ============================================================


def show_prediction_result(
    result: dict,
    input_data: pd.DataFrame | None = None,
) -> None:
    """Display ensemble prediction with per-model confidence breakdown.

    Works with the dict returned by ``make_prediction()``:
        predictions, probabilities, threshold, model_breakdown, errors

    ``probabilities`` are P(class=1), where class 1 = good credit / low risk.
    """
    if result["errors"]:
        st.error(f"Prediction failed: {result['errors']}")
        return

    good_proba = float(result["probabilities"][0])
    bad_proba = probability_of_bad(good_proba)
    decision = result["predictions"][0]
    threshold = result["threshold"]

    # ── Main verdict ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Prediction Result")

    col_verdict, col_detail = st.columns([1, 2])

    with col_verdict:
        if decision == BAD_CLASS:
            st.error("⚠️ **HIGH RISK**")
        else:
            st.success("✅ **LOW RISK**")
        st.metric("Good Credit Probability", f"{good_proba:.1%}")
        st.metric("Default Risk", f"{bad_proba:.1%}")

    with col_detail:
        st.caption(
            f"Decision threshold: {threshold:.2f} · "
            f"Probabilities above this threshold are classified as low risk."
        )
        st.caption(
            "Displayed model probabilities are P(good credit); "
            "default risk is computed as 1 − P(good credit)."
        )

        # ── Per-model breakdown (config-driven) ──────────────────────────
        st.markdown("**Model Breakdown:**")
        breakdown = result.get("model_breakdown", {})

        for model_key in config.models:
            model_proba = breakdown.get(f"{model_key}_proba", [None])[0]
            weight = config.ensemble.weights[model_key]
            label = MODEL_LABELS.get(model_key, model_key.upper())
            if model_proba is not None:
                model_good_proba = float(model_proba)
                model_bad_proba = probability_of_bad(model_good_proba)
                risk_tag = risk_status_from_probability(
                    model_good_proba,
                    threshold,
                )
                st.progress(
                    model_good_proba,
                    text=(
                        f"{label}: good {model_good_proba:.1%} · "
                        f"default {model_bad_proba:.1%} · "
                        f"{risk_tag} · weight {weight:.1f}"
                    ),
                )
            else:
                st.text(f"{label}: N/A")

        st.markdown("---")
        st.progress(
            good_proba,
            text=(
                f"Ensemble: good {good_proba:.1%} · "
                f"default {bad_proba:.1%} · "
                f"{risk_status_from_probability(good_proba, threshold)}"
            ),
        )

    # ── Optional input-data summary ───────────────────────────────────────
    if input_data is not None:
        with st.expander("View Input Data"):
            display_df = input_data.T.reset_index()
            display_df.columns = ["Feature", "Value"]
            display_df["Feature"] = display_df["Feature"].map(lambda x: FEATURE_LABELS.get(x, x))
            st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================================
# TABS
# ============================================================


def tab_random_samples():
    """Tab 1: Sample random applicants from the test set."""
    st.header("🎲 Test with Random Samples")
    st.write(
        "Click a button to load a random sample from the test dataset. "
        "Click again for a different sample!"
    )

    sample_df = load_sample_data()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            "🟢 Good Credit Sample",
            use_container_width=True,
            help="Randomly sample an applicant labeled as good risk",
        ):
            sample = sample_df[sample_df["class"] == GOOD_CLASS].sample(1)
            st.session_state["sample"] = sample.drop(columns=["class"])
            st.session_state["true_label"] = GOOD_CLASS
            st.session_state["sample_type"] = "Good Risk"

    with col2:
        if st.button(
            "🔴 Bad Credit Sample",
            use_container_width=True,
            help="Randomly sample an applicant labeled as bad risk",
        ):
            sample = sample_df[sample_df["class"] == BAD_CLASS].sample(1)
            st.session_state["sample"] = sample.drop(columns=["class"])
            st.session_state["true_label"] = BAD_CLASS
            st.session_state["sample_type"] = "Bad Risk"

    with col3:
        if st.button(
            "🎯 Random Sample",
            use_container_width=True,
            help="Randomly sample any applicant",
        ):
            sample = sample_df.sample(1)
            st.session_state["sample"] = sample.drop(columns=["class"])
            st.session_state["true_label"] = sample["class"].values[0]
            st.session_state["sample_type"] = "Random"

    if "sample" not in st.session_state:
        return

    input_data = st.session_state["sample"]

    # ── Sample info badges ────────────────────────────────────────────────
    st.markdown("---")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"📋 **Sample Type:** {st.session_state['sample_type']}")
    with col_info2:
        true_label = st.session_state["true_label"]
        human_label = label_text(true_label)
        if true_label == BAD_CLASS:
            st.warning(f"🏷️ **True Label:** {human_label}")
        else:
            st.success(f"🏷️ **True Label:** {human_label}")

    # ── Predict ───────────────────────────────────────────────────────────
    pipelines = get_pipelines()
    result = make_prediction(input_data=input_data, pipelines=pipelines)
    show_prediction_result(result, input_data)


def tab_manual_input():
    """Tab 2: Manual applicant data entry form."""
    st.header("✏️ Manual Input")
    st.write("Enter applicant details to get a credit risk prediction.")

    with st.expander("Open Input Form", expanded=True):
        input_data: dict = {}

        # ── Categorical features ──────────────────────────────────────────
        col1, col2 = st.columns(2)
        feature_list = list(CATEGORICAL_FEATURES.keys())
        half = len(feature_list) // 2

        with col1:
            st.markdown("**Categorical Features**")
            for feature in feature_list[:half]:
                label = FEATURE_LABELS.get(feature, feature)
                options = CATEGORICAL_FEATURES[feature]
                help_text = FEATURE_HELP.get(feature)
                input_data[feature] = st.selectbox(
                    label,
                    options,
                    key=f"cat_{feature}",
                    help=help_text,
                )

        with col2:
            st.markdown("&nbsp;")  # spacer to align with left column header
            for feature in feature_list[half:]:
                label = FEATURE_LABELS.get(feature, feature)
                options = CATEGORICAL_FEATURES[feature]
                help_text = FEATURE_HELP.get(feature)
                input_data[feature] = st.selectbox(
                    label,
                    options,
                    key=f"cat_{feature}",
                    help=help_text,
                )

        st.markdown("---")

        # ── Numeric features ──────────────────────────────────────────────
        col3, col4 = st.columns(2)
        numeric_list = list(NUMERIC_FEATURES.keys())
        half_num = len(numeric_list) // 2

        with col3:
            st.markdown("**Numeric Features**")
            for feature in numeric_list[:half_num]:
                label = FEATURE_LABELS.get(feature, feature)
                feat_cfg = NUMERIC_FEATURES[feature]
                help_text = FEATURE_HELP.get(feature)
                input_data[feature] = st.number_input(
                    label,
                    min_value=feat_cfg["min"],
                    max_value=feat_cfg["max"],
                    value=feat_cfg["default"],
                    step=feat_cfg["step"],
                    key=f"num_{feature}",
                    help=help_text,
                )

        with col4:
            st.markdown("&nbsp;")
            for feature in numeric_list[half_num:]:
                label = FEATURE_LABELS.get(feature, feature)
                feat_cfg = NUMERIC_FEATURES[feature]
                help_text = FEATURE_HELP.get(feature)
                input_data[feature] = st.number_input(
                    label,
                    min_value=feat_cfg["min"],
                    max_value=feat_cfg["max"],
                    value=feat_cfg["default"],
                    step=feat_cfg["step"],
                    key=f"num_{feature}",
                    help=help_text,
                )

        # ── Predict ───────────────────────────────────────────────────────
        if st.button(
            "🔮 Predict Risk",
            type="primary",
            use_container_width=True,
        ):
            input_df = pd.DataFrame([input_data])
            pipelines = get_pipelines()
            result = make_prediction(input_data=input_df, pipelines=pipelines)
            show_prediction_result(result, input_df)


# ============================================================
# MAIN
# ============================================================


def main():
    st.set_page_config(
        page_title="Credit Risk Assessment",
        page_icon="🏦",
        layout="wide",
    )

    st.title("🏦 Credit Risk Assessment")
    st.caption(
        f"Ensemble: {' + '.join(k.upper() for k in config.models)} · "
        f"Threshold: {config.ensemble.threshold:.2f} · "
        f"Cost ratio: FP={config.cost_matrix.false_positive:.0f}× FN"
    )

    # ── Cost-sensitive explainer ──────────────────────────────────────────
    with st.expander("ℹ️ About the Model's Cost-Sensitive Approach"):
        st.markdown(
            """
            This model is optimised for **minimising business cost**,
            not just accuracy.

            According to the
            [German Credit
            Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
            documentation:

            | Prediction | Actual | Cost |
            |------------|--------|------|
            | Good Risk | Bad Risk (FP) | **5** |
            | Bad Risk | Good Risk (FN) | **1** |

            **What this means:** It's 5× more costly to approve a loan
            for someone who will default than to reject a creditworthy
            applicant. The model is intentionally **conservative** —
            it may classify some good-risk applicants
            as bad-risk to avoid the larger cost of defaults.
            """
        )

    # ── Load models ───────────────────────────────────────────────────────
    try:
        get_pipelines()  # warm the cache; errors surface here
    except Exception as exc:
        st.error(f"❌ Could not load models: {exc}")
        st.stop()

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["🎲 Random Samples", "✏️ Manual Input"])
    with tab1:
        tab_random_samples()
    with tab2:
        tab_manual_input()

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption("Built with Streamlit · [View Source Code](https://github.com/ntinasf/credit-risk)")


if __name__ == "__main__":
    main()
