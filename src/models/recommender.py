"""SKU recommender using implicit ALS with popularity fallback."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from src.config import RANDOM_STATE


@dataclass
class RecommenderResult:
    model: AlternatingLeastSquares
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_index: dict
    item_index: dict
    popularity: np.ndarray


def _build_matrix(interactions: pd.DataFrame):
    user_ids = interactions["cliente_id"].unique()
    item_ids = interactions["produto_id"].unique()
    ui = {u: i for i, u in enumerate(user_ids)}
    ii = {p: i for i, p in enumerate(item_ids)}
    rows = interactions["cliente_id"].map(ui).to_numpy()
    cols = interactions["produto_id"].map(ii).to_numpy()
    vals = interactions["n"].astype(float).to_numpy()
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids))), user_ids, item_ids


def train_recommender(interactions: pd.DataFrame, factors: int = 64, iterations: int = 20) -> RecommenderResult:
    matrix, user_ids, item_ids = _build_matrix(interactions)
    model = AlternatingLeastSquares(factors=factors, iterations=iterations, regularization=0.05,
                                    random_state=RANDOM_STATE, use_gpu=False)
    model.fit(matrix)
    popularity = np.asarray(matrix.sum(axis=0)).ravel()
    return RecommenderResult(model=model, user_ids=user_ids, item_ids=item_ids,
                             user_index={u: i for i, u in enumerate(user_ids)},
                             item_index={p: i for i, p in enumerate(item_ids)}, popularity=popularity)


def recommend(result: RecommenderResult, user_id, n: int = 10):
    if user_id not in result.user_index:
        top = np.argsort(result.popularity)[::-1][:n]
        return [(result.item_ids[i], float(result.popularity[i])) for i in top]
    u = result.user_index[user_id]
    # build a CSR user-items matrix (implicit >=0.7 requires sparse input)
    user_items = sp.csr_matrix(result.model.user_factors[u].reshape(1, -1))
    ids, scores = result.model.recommend(u, user_items, N=n, filter_already_liked_items=True)
    return [(result.item_ids[i], float(s)) for i, s in zip(ids, scores, strict=False)]


def evaluate_at_k(result: RecommenderResult, interactions: pd.DataFrame,
                  k: int = 10, sample_users: int | None = 500,
                  seed: int = 42) -> dict:
    """Leave-one-out recall@k.

    For each sampled user we hide one of their purchased items, rebuild that
    user's interaction vector without it, and ask the model to rank top-k.
    The previous "last item per user" proxy held out the user's smallest
    purchase, which the model never ranked highly and produced a misleading 0.015.
    """
    rng = np.random.default_rng(seed)
    matrix, user_ids, item_ids = _build_matrix(interactions)
    ui = {u: i for i, u in enumerate(user_ids)}

    users = list(user_ids)
    if sample_users and len(users) > sample_users:
        users = rng.choice(users, size=sample_users, replace=False).tolist()

    hits, total = 0, 0
    for u in users:
        if u not in ui:
            continue
        idx = ui[u]
        row = matrix.getrow(idx).toarray().ravel()
        nz = np.where(row > 0)[0]
        if len(nz) < 2:
            continue
        held = int(rng.choice(nz))
        train = row.copy()
        train[held] = 0
        user_items = sp.csr_matrix(train.reshape(1, -1))
        ids, _ = result.model.recommend(idx, user_items, N=k,
                                        filter_already_liked_items=True)
        hits += int(held in set(ids))
        total += 1
    return {"recall_at_k": hits / total if total else 0.0, "k": k,
            "n_users": total, "n_sampled": len(users)}

