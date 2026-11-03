"""Churn classifier: calibrated LightGBM in a sklearn pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE

# The churn label is a revenue-decline rule based on receita_6m vs
# receita_6m_anterior. Those two columns are excluded from the feature set to
# avoid target leakage; the model learns from RFM behaviour instead.
FEATURES = ["recencia_dias", "frequencia", "ticket_medio", "receita_total", "skus_distintos"]
CATEGORICAL = ["segmento", "uf"]
TARGET = "churn"


@dataclass
class ChurnResult:
    auc: float
    pr_auc: float
    brier: float
    model: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    y_proba: np.ndarray


def _build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    base = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    return Pipeline([("pre", pre), ("clf", CalibratedClassifierCV(base, method="sigmoid", cv=3))])


def train_churn(df: pd.DataFrame, test_size: float = 0.25) -> ChurnResult:
    X = df[FEATURES + CATEGORICAL]
    y = df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    pipe = _build_pipeline().fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    return ChurnResult(auc=roc_auc_score(y_te, proba), pr_auc=average_precision_score(y_te, proba),
                       brier=brier_score_loss(y_te, proba), model=pipe, X_test=X_te, y_test=y_te, y_proba=proba)

