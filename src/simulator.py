"""Reward simulator: Logistic Regression acts as the conversion environment.

Expected reward per arm: cellular/telephone = P(convert) * channel_lift, skip = 0.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ARMS = ["cellular", "telephone", "skip"]


def channel_lift(df: pd.DataFrame):
    """Per-channel conversion lift vs the global conversion rate."""
    base_rate = float(df["y"].mean())
    lifts = {}
    for ch in ["cellular", "telephone"]:
        sub = df[df["contact"] == ch]
        lifts[ch] = float(sub["y"].mean() / base_rate) if len(sub) and base_rate > 0 else 1.0
    return base_rate, lifts


class RewardSimulator:
    def __init__(self, model=None, seed: int = 42):
        self.model = model or LogisticRegression(max_iter=1000)
        self.rng = np.random.default_rng(seed)
        self.base_rate_ = None
        self.lifts_ = None

    def fit(self, X_train, y_train, df_raw=None):
        self.model.fit(X_train, y_train)
        self.base_rate_, self.lifts_ = (
            channel_lift(df_raw)
            if df_raw is not None
            else (float(np.mean(y_train)), {a: 1.0 for a in ["cellular", "telephone"]})
        )
        return self

    def propensity(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def expected_reward(self, X, arm: str) -> np.ndarray:
        p = self.propensity(X)
        if arm == "skip":
            return np.zeros(len(p))
        return np.clip(p * self.lifts_.get(arm, 1.0), 0.0, 1.0)  # P*lift can exceed 1

    def sample_reward(self, X, arm: str, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        return rng.binomial(1, self.expected_reward(X, arm))
