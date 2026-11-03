"""Customer segmentation: KMeans (elbow + silhouette) and DBSCAN comparison."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE

FEATURES = ["recencia", "frequencia", "monetario", "n_skus"]


@dataclass
class SegmentResult:
    labels: np.ndarray
    k: int
    silhouette: float
    davies_bouldin: float
    profile: pd.DataFrame
    model: KMeans


def _scale(df: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(df[FEATURES].astype(float))


def choose_k(X: np.ndarray, k_min: int = 2, k_max: int = 8) -> tuple[int, dict[int, float]]:
    scores = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        scores[k] = silhouette_score(X, km.labels_)
    return max(scores, key=scores.get), scores


def fit_segmentation(df: pd.DataFrame, k: int | None = None) -> SegmentResult:
    X = _scale(df)
    if k is None:
        k, _ = choose_k(X)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
    profile = df.assign(cluster=km.labels_).groupby("cluster")[FEATURES].mean()
    profile["size"] = df.assign(cluster=km.labels_).groupby("cluster").size()
    return SegmentResult(labels=km.labels_, k=k, silhouette=float(silhouette_score(X, km.labels_)),
                         davies_bouldin=float(davies_bouldin_score(X, km.labels_)), profile=profile, model=km)


def fit_dbscan(df: pd.DataFrame, eps: float = 1.2, min_samples: int = 5) -> dict:
    X = _scale(df)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return {"labels": labels, "n_clusters": n_clusters, "noise_ratio": float((labels == -1).mean())}

