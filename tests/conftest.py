import pandas as pd
import pytest

from credit_risk_model.config.core import DATA_DIR, config
from credit_risk_model.processing.features import FeatureEngineer


@pytest.fixture(scope="session")
def raw_sample() -> pd.DataFrame:
    """10 raw rows from the processed CSV — representative inputs."""
    return pd.read_csv(DATA_DIR / config.training_data_file).head(10)


@pytest.fixture(scope="session")
def engineered_sample(raw_sample) -> pd.DataFrame:
    """10 rows after FeatureEngineer.transform() with default settings."""
    fe = FeatureEngineer()
    return fe.transform(raw_sample.drop(columns=[config.target]))


@pytest.fixture(scope="session")
def X_train_small() -> pd.DataFrame:
    """Small stratified training slice for fast pipeline fit tests.

    Uses stratified sampling so both classes are present — WOE
    encoder requires a binary target with values {0, 1}.
    """
    df = pd.read_csv(DATA_DIR / config.training_data_file)
    pos = df[df[config.target] == 1].head(100)
    neg = df[df[config.target] == 0].head(100)
    sample = pd.concat([pos, neg], ignore_index=True)
    return sample.drop(columns=[config.target])


@pytest.fixture(scope="session")
def y_train_small() -> pd.Series:
    df = pd.read_csv(DATA_DIR / config.training_data_file)
    pos = df[df[config.target] == 1].head(100)
    neg = df[df[config.target] == 0].head(100)
    sample = pd.concat([pos, neg], ignore_index=True)
    return sample[config.target]


@pytest.fixture(scope="session")
def mock_pipelines(X_train_small, y_train_small):
    """Fit lightweight versions of all four pipelines for integration tests.

    Uses default (non-tuned) estimators to keep tests fast.
    This fixture is session-scoped so the fit runs only once per test session.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    from credit_risk_model.processing.preprocessors import build_catboost_pipeline, build_pipeline

    pipelines = {}
    pipelines["lrc"] = build_pipeline(
        LogisticRegression(max_iter=200, random_state=8),
        config.models["lrc"],
    )
    pipelines["rfc"] = build_pipeline(
        RandomForestClassifier(n_estimators=10, random_state=8),
        config.models["rfc"],
    )
    pipelines["svc"] = build_pipeline(
        SVC(probability=True, random_state=8),
        config.models["svc"],
    )
    pipelines["cat"] = build_catboost_pipeline(
        config.models["cat"],
        random_state=8,
        scale_pos_weight=5.0,
    )
    for pipeline in pipelines.values():
        pipeline.fit(X_train_small, y_train_small)
    return pipelines
