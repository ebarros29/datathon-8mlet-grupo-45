"""Contextual Thompson Sampling over offer arms + deterministic baseline.

Each (arm, propensity-segment) keeps a Beta(1,1) posterior. Per client we sample a
conversion rate per arm from its segment posterior, subtract the contact cost for
contact arms (skip scores 0), and pick the highest score. Context enters via the
per-segment posteriors; exploration happens early, then the policy exploits.
"""
import numpy as np

from src.simulator import ARMS, RewardSimulator


def replay_baseline(simulator: RewardSimulator, X_eval, arm: str = "cellular", seed: int = 42) -> np.ndarray:
    """Fixed rule: always present ``arm`` to every client."""
    return simulator.sample_reward(X_eval, arm, seed=seed)


class ContextualThompsonSampling:
    def __init__(self, arms=ARMS, n_segments: int = 3, contact_cost: float = 0.07,
                 prior_alpha: float = 1.0, prior_beta: float = 1.0, seed: int = 42):
        self.arms = list(arms)
        self.n_segments = n_segments
        self.contact_cost = contact_cost
        self.alpha = {(a, s): prior_alpha for a in self.arms for s in range(n_segments)}
        self.beta = {(a, s): prior_beta for a in self.arms for s in range(n_segments)}
        self.counts = {(a, s): 0 for a in self.arms for s in range(n_segments)}
        self.rewards = {(a, s): 0.0 for a in self.arms for s in range(n_segments)}
        self.thresholds_ = None
        self.rng = np.random.default_rng(seed)

    def fit_segments(self, train_propensity: np.ndarray):
        self.thresholds_ = np.quantile(
            train_propensity, [1 / self.n_segments * (k + 1) for k in range(self.n_segments - 1)]
        )
        return self

    def segment_of(self, propensity: float) -> int:
        assert self.thresholds_ is not None, "call fit_segments() first"
        return int(np.digitize(propensity, self.thresholds_))

    def choose_arm(self, propensity: float) -> str:
        seg = self.segment_of(propensity)
        scores = {}
        for a in self.arms:
            s = self.rng.beta(self.alpha[(a, seg)], self.beta[(a, seg)])
            scores[a] = s - self.contact_cost if a != "skip" else 0.0
        return max(scores, key=scores.get)

    def update(self, arm: str, seg: int, reward: float) -> None:
        self.counts[(arm, seg)] += 1
        self.rewards[(arm, seg)] += float(reward)
        self.alpha[(arm, seg)] += float(reward)
        self.beta[(arm, seg)] += 1.0 - float(reward)

    def conversion_rate(self, arm: str, seg: int) -> float:
        return self.alpha[(arm, seg)] / (self.alpha[(arm, seg)] + self.beta[(arm, seg)])


def replay_contextual_bandit(simulator, X_eval, bandit):
    p = simulator.propensity(X_eval)
    chosen, rewards, segs = [], [], []
    for i in range(len(X_eval)):
        seg = bandit.segment_of(p[i])
        arm = bandit.choose_arm(p[i])
        reward = float(simulator.sample_reward(X_eval[i:i + 1], arm, seed=None)[0])
        bandit.update(arm, seg, reward)
        chosen.append(arm)
        rewards.append(reward)
        segs.append(seg)
    return chosen, rewards, segs


def recommend(simulator, bandit, preprocessor, X_row) -> dict:
    """Exploit: recommend the arm with the highest client-specific expected reward net of cost."""
    Xt = preprocessor.transform(X_row)
    p = float(simulator.propensity(Xt)[0])
    seg = bandit.segment_of(p)
    exp = {a: float(simulator.expected_reward(Xt, a)[0]) for a in bandit.arms}
    score = {a: exp[a] - bandit.contact_cost if a != "skip" else 0.0 for a in bandit.arms}
    offer = max(score, key=score.get)

    if offer == "skip":
        best = max(exp, key=exp.get)
        rationale = f"p={p:.2f}; best contact '{best}' ({exp[best]:.2f}) < cost {bandit.contact_cost:.2f} -> skip"
    else:
        runner_up = max(exp, key=lambda a: exp[a] if a != offer else -1)
        rationale = f"p={p:.2f}; '{offer}' ({exp[offer]:.2f}) beats '{runner_up}' ({exp[runner_up]:.2f}) net of cost {bandit.contact_cost:.2f}"

    return {
        "recommended_offer": offer,
        "propensity": round(p, 4),
        "segment": int(seg),
        "expected_reward": {a: round(v, 4) for a, v in exp.items()},
        "rationale": rationale,
    }
