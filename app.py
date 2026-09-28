"""FastAPI service: returns the recommended offer for a client.

Run: uvicorn app:app --reload  (POST /recommend, see README for the curl example)
"""
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.features import FEATURE_COLUMNS, build_preprocessor, clean_data, load_data, split_data
from src.simulator import RewardSimulator
from src.bandit import ContextualThompsonSampling, recommend

CONTACT_COST = 0.07

app = FastAPI(title="Adaptive Offer Recommendation", version="1.0.0")


def _build_pipeline():
    """Train the simulator + bandit once at startup."""
    df = clean_data(load_data())
    X, y = df[FEATURE_COLUMNS], df["y"]
    X_train, _, y_train, _ = split_data(X, y)
    pre = build_preprocessor().fit(X_train)
    X_train_t = pre.transform(X_train)
    sim = RewardSimulator().fit(X_train_t, y_train, df_raw=df)
    bandit = ContextualThompsonSampling(contact_cost=CONTACT_COST).fit_segments(sim.propensity(X_train_t))
    return pre, sim, bandit


PRE, SIM, BANDIT = _build_pipeline()


class ClientFeatures(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    balance: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str


@app.get("/")
def root():
    return {"service": "Adaptive Offer Recommendation", "arms": BANDIT.arms, "contact_cost": BANDIT.contact_cost}


@app.post("/recommend")
def get_recommendation(client: ClientFeatures):
    row = pd.DataFrame([client.model_dump()])[FEATURE_COLUMNS]
    rec = recommend(SIM, BANDIT, PRE, row)
    rec["message"] = f"Recommended offer: {rec['recommended_offer']}"
    return rec
