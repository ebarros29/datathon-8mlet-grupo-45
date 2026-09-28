"""Feature prep for the bank-marketing dataset (Kaggle: henriqueyamahata/bank-marketing).

Leakage dropped: ``duration`` (known only after the call), ``day``/``month`` (campaign
period). ``contact`` is the action (arm), not a feature.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bank.csv"

TARGET_COLUMN = "y"
ACTION_COLUMN = "contact"

DROP_COLUMNS = ["duration", "day", "month"]

CATEGORICAL_COLUMNS = ["job", "marital", "education", "default", "housing", "loan", "poutcome"]
NUMERIC_COLUMNS = ["age", "balance", "campaign", "pdays", "previous"]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leakage columns; target -> 0/1."""
    df = df.copy().drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "yes").astype(int)
    return df


def build_preprocessor(categorical_columns=CATEGORICAL_COLUMNS, numeric_columns=NUMERIC_COLUMNS):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical_columns)),
            ("num", StandardScaler(), list(numeric_columns)),
        ]
    )


def split_data(X, y, test_size: float = 0.3, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
