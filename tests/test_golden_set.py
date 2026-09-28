"""Golden Set: 5 fixed clients; recommendations must be valid, deterministic and sensible."""
import pytest
import pandas as pd

from src.features import FEATURE_COLUMNS, build_preprocessor, clean_data, load_data, split_data
from src.simulator import ARMS, RewardSimulator
from src.bandit import ContextualThompsonSampling, recommend

GOLDEN_CLIENTS = [
    {"age": 34, "job": "management", "marital": "married", "education": "tertiary",
     "default": "no", "housing": "yes", "loan": "no", "balance": 2500,
     "campaign": 1, "pdays": -1, "previous": 1, "poutcome": "success"},
    {"age": 41, "job": "blue-collar", "marital": "married", "education": "secondary",
     "default": "no", "housing": "yes", "loan": "no", "balance": 900,
     "campaign": 2, "pdays": -1, "previous": 0, "poutcome": "unknown"},
    {"age": 58, "job": "retired", "marital": "married", "education": "primary",
     "default": "no", "housing": "yes", "loan": "no", "balance": 120,
     "campaign": 3, "pdays": 90, "previous": 2, "poutcome": "failure"},
    {"age": 29, "job": "technician", "marital": "single", "education": "secondary",
     "default": "no", "housing": "no", "loan": "no", "balance": 1800,
     "campaign": 1, "pdays": 6, "previous": 1, "poutcome": "success"},
    {"age": 47, "job": "services", "marital": "divorced", "education": "secondary",
     "default": "no", "housing": "yes", "loan": "yes", "balance": 300,
     "campaign": 4, "pdays": -1, "previous": 0, "poutcome": "unknown"},
]


@pytest.fixture(scope="module")
def pipeline():
    df = clean_data(load_data())
    X, y = df[FEATURE_COLUMNS], df["y"]
    X_train, _, y_train, _ = split_data(X, y)
    pre = build_preprocessor().fit(X_train)
    X_train_t = pre.transform(X_train)
    sim = RewardSimulator().fit(X_train_t, y_train, df_raw=df)
    bandit = ContextualThompsonSampling(contact_cost=0.07).fit_segments(sim.propensity(X_train_t))
    return pre, sim, bandit


def _recommend(pre, sim, bandit, client):
    return recommend(sim, bandit, pre, pd.DataFrame([client])[FEATURE_COLUMNS])


def test_valid_and_deterministic(pipeline):
    pre, sim, bandit = pipeline
    for client in GOLDEN_CLIENTS:
        rec = _recommend(pre, sim, bandit, client)
        assert rec["recommended_offer"] in ARMS
        assert rec["rationale"]
        assert rec["recommended_offer"] == _recommend(pre, sim, bandit, client)["recommended_offer"]


def test_high_propensity_clients_are_contacted(pipeline):
    pre, sim, bandit = pipeline
    for name in ["management", "technician"]:
        client = next(c for c in GOLDEN_CLIENTS if c["job"] == name)
        rec = _recommend(pre, sim, bandit, client)
        assert rec["propensity"] > 0.3
        assert rec["recommended_offer"] != "skip"


def test_lowest_propensity_client_is_skipped(pipeline):
    pre, sim, bandit = pipeline
    client = next(c for c in GOLDEN_CLIENTS if c["job"] == "services")
    rec = _recommend(pre, sim, bandit, client)
    assert rec["propensity"] < 0.1
    assert rec["recommended_offer"] == "skip"
    assert rec["expected_reward"]["skip"] == 0.0


def test_expected_rewards_are_probabilities(pipeline):
    pre, sim, bandit = pipeline
    for client in GOLDEN_CLIENTS:
        rec = _recommend(pre, sim, bandit, client)
        for arm, value in rec["expected_reward"].items():
            assert 0.0 <= value <= 1.0
