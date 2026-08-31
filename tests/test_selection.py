"""Tests for the ensemble selection step.

`write_config` rewrites model_config.yml with a regex, which is the kind of thing
that silently corrupts a file. These tests pin down that it replaces only the
ensemble block, preserves comments, and is idempotent.
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

from credit_risk_model.config.core import AppConfig, config

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location("select_ensemble", SCRIPTS / "select_ensemble.py")
select_ensemble = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(select_ensemble)


SAMPLE_CONFIG = """\
# leading comment
target: class

# -- ensemble ------------------------
# a comment above the block
ensemble:
  threshold: 0.5
  weights:
    lrc: 1.0
    rfc: 1.0
    svc: 1.0
    cat: 1.0

# -- trailing section ----------------
mlflow:
  experiment_name: demo
"""


@pytest.fixture
def config_file(tmp_path) -> Path:
    path = tmp_path / "model_config.yml"
    path.write_text(SAMPLE_CONFIG)
    return path


def test_write_config_updates_weights_and_threshold(config_file):
    weights = {"lrc": 2.0, "rfc": 2.5, "svc": 1.5, "cat": 1.0}
    select_ensemble.write_config(weights, 0.71, path=config_file)

    parsed = yaml.safe_load(config_file.read_text())
    assert parsed["ensemble"]["threshold"] == 0.71
    assert parsed["ensemble"]["weights"] == weights


def test_write_config_leaves_the_rest_of_the_file_intact(config_file):
    select_ensemble.write_config(
        {"lrc": 1.0, "rfc": 1.0, "svc": 1.0, "cat": 1.0}, 0.6, path=config_file
    )
    text = config_file.read_text()

    assert "# leading comment" in text
    assert "# a comment above the block" in text
    assert "# -- trailing section ----------------" in text
    assert yaml.safe_load(text)["mlflow"]["experiment_name"] == "demo"
    assert yaml.safe_load(text)["target"] == "class"


def test_write_config_is_idempotent(config_file):
    weights = {"lrc": 1.5, "rfc": 2.0, "svc": 1.0, "cat": 2.5}
    select_ensemble.write_config(weights, 0.68, path=config_file)
    once = config_file.read_text()
    select_ensemble.write_config(weights, 0.68, path=config_file)
    assert config_file.read_text() == once


def test_write_config_raises_when_block_is_missing(tmp_path):
    path = tmp_path / "no_ensemble.yml"
    path.write_text("target: class\n")
    complete = {k: 1.0 for k in select_ensemble.MODEL_ORDER}
    with pytest.raises(RuntimeError, match="ensemble"):
        select_ensemble.write_config(complete, 0.5, path=path)


def test_write_config_rejects_incomplete_weights(config_file):
    with pytest.raises(ValueError, match="missing entries"):
        select_ensemble.write_config({"lrc": 1.0}, 0.5, path=config_file)


def test_written_config_still_passes_pydantic_validation(config_file):
    """A written file must survive the real AppConfig validators."""
    weights = {k: 1.0 for k in select_ensemble.MODEL_ORDER}
    select_ensemble.write_config(weights, 0.66, path=config_file)
    written = yaml.safe_load(config_file.read_text())["ensemble"]

    # Rebuild a full config document using the shipped models section.
    raw = {
        "training_data_file": config.training_data_file,
        "test_data_file": config.test_data_file,
        "target": config.target,
        "random_state": config.random_state,
        "val_size": config.val_size,
        "cost_matrix": config.cost_matrix.model_dump(),
        "ensemble": written,
        "models": {k: v.model_dump() for k, v in config.models.items()},
        "mlflow": config.mlflow,
    }
    rebuilt = AppConfig(**raw)
    assert rebuilt.ensemble.threshold == 0.66


def test_shipped_config_matches_the_scripts_model_order():
    """Guards against a model being added to config but not to the sweep."""
    assert set(select_ensemble.MODEL_ORDER) == set(config.models.keys())
