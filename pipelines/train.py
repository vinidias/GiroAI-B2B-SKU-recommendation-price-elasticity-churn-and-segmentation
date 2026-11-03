"""End-to-end training pipeline with MLflow tracking.

Run:
    python -m pipelines.train

This seeds the data (if missing), validates it, builds features, trains the
four models and logs metrics/artifacts to MLflow.
"""
from __future__ import annotations

import json
import logging

import mlflow
import pandas as pd

from src.config import DATA_DIR, MLRUNS_DIR, RANDOM_STATE, RAW_FILES
from src.data.synthetic import generate, write_to_disk
from src.data.validate import load_and_validate_all
from src.experiments.ab_test import simulate_experiment
from src.features.build import (build_churn_features, build_elasticity_features,
                                build_interactions, build_segmentation_features)
from src.models.churn import train_churn
from src.models.elasticity import fit_elasticity
from src.models.recommender import evaluate_at_k, train_recommender
from src.models.segmentation import fit_dbscan, fit_segmentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("giro.train")


def ensure_data() -> None:
    missing = [f for f in RAW_FILES.values() if not (DATA_DIR / f).exists()]
    if missing:
        log.info("Missing raw files %s — generating synthetic fallback.", missing)
        write_to_disk(generate())
    else:
        log.info("Raw data present in %s", DATA_DIR)


def main() -> None:
    ensure_data()
    tables = load_and_validate_all()
    log.info("Validated tables: %s", {k: len(v) for k, v in tables.items()})

    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    mlflow.set_experiment("giro-ai")

    with mlflow.start_run(run_name="full-pipeline"):
        # --- Elasticity ---
        el_df = build_elasticity_features(tables["pedidos"], tables["itens_pedido"],
                                         tables["produtos"], tables["clientes"])
        el_model, el_table = fit_elasticity(el_df, segment_col="categoria")
        mlflow.log_metric("elasticity_r2", float(el_model.rsquared))
        el_table.to_csv("data/processed/elasticity_by_category.csv", index=False)
        mlflow.log_artifact("data/processed/elasticity_by_category.csv")
        log.info("Elasticity R2=%.3f\n%s", el_model.rsquared, el_table.to_string(index=False))

        # --- Churn ---
        ch_df = build_churn_features(tables["clientes"], tables["pedidos"], tables["itens_pedido"])
        ch = train_churn(ch_df)
        mlflow.log_metrics({"churn_auc": ch.auc, "churn_pr_auc": ch.pr_auc, "churn_brier": ch.brier})
        log.info("Churn AUC=%.3f PR-AUC=%.3f Brier=%.3f", ch.auc, ch.pr_auc, ch.brier)

        # --- Segmentation ---
        seg_df = build_segmentation_features(tables["clientes"], tables["pedidos"], tables["itens_pedido"])
        seg = fit_segmentation(seg_df)
        db = fit_dbscan(seg_df)
        mlflow.log_metrics({
            "seg_k": seg.k, "seg_silhouette": seg.silhouette,
            "seg_davies_bouldin": seg.davies_bouldin,
            "seg_dbscan_clusters": db["n_clusters"], "seg_dbscan_noise": db["noise_ratio"],
        })
        seg.profile.to_csv("data/processed/segment_profile.csv")
        mlflow.log_artifact("data/processed/segment_profile.csv")
        log.info("Segmentation k=%d silhouette=%.3f", seg.k, seg.silhouette)

        # --- Recommender ---
        inter = build_interactions(tables["pedidos"], tables["itens_pedido"])
        rec = train_recommender(inter)
        rec_metrics = evaluate_at_k(rec, inter, k=10)
        mlflow.log_metrics({"recall_at_10": rec_metrics["recall_at_k"]})
        log.info("Recommender recall@10=%.3f", rec_metrics["recall_at_k"])

        # --- A/B simulation ---
        ab = simulate_experiment(tables["pedidos"], tables["itens_pedido"], uplift=0.05)
        r = ab["result"]
        mlflow.log_metrics({"ab_p_value": r.p_value, "ab_uplift_pct": r.uplift_pct})
        log.info("A/B uplift=%.2f%% p=%.4f significant=%s", r.uplift_pct, r.p_value, r.significant)

        summary = {
            "elasticity_r2": float(el_model.rsquared),
            "churn_auc": ch.auc, "churn_pr_auc": ch.pr_auc,
            "seg_k": seg.k, "seg_silhouette": seg.silhouette,
            "recommender_recall_at_10": rec_metrics["recall_at_k"],
            "ab_p_value": r.p_value, "ab_uplift_pct": r.uplift_pct,
        }
        with open("data/processed/run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact("data/processed/run_summary.json")
        log.info("Pipeline done: %s", summary)


if __name__ == "__main__":
    main()
