# Giro AI

> End-to-end data science platform for B2B food distribution: SKU recommendation, price elasticity, churn prediction and customer segmentation — built with CRISP-DM, MLflow, DVC-ready data versioning and A/B testing.

## Business impact

| Question | Model | Gain |
|---|---|---|
| What should we recommend to this customer? | Implicit ALS collaborative filtering | Higher AOV, cross-sell revenue |
| How will demand react if we change the price? | Log-log OLS per category | Margin-aware pricing |
| Which customers are about to churn? | Calibrated LightGBM | Proactive retention |
| Which customers behave alike? | KMeans / DBSCAN | Targeted CRM and territory design |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e ".[dev]"

# Generate fallback synthetic dataset (Atlas layout)
python -m src.data.synthetic

# Run full pipeline (trains all four models, logs to MLflow)
python -m pipelines.train

# Inspect results
mlflow ui --backend-store-uri ./mlruns
```

## Using the real Atacado Atlas dataset

Drop the CSVs into `data/raw/` with these names:

```
data/raw/clientes.csv
data/raw/produtos.csv
data/raw/pedidos.csv
data/raw/itens_pedido.csv
data/raw/precos_produto.csv
data/raw/vendedores.csv
```

The loader (`src/data/validate.py`) validates them against Pandera schemas
and the pipeline runs unchanged. Remove the fallback files first or the
synthetic generator will skip if CSVs already exist.

## CRISP-DM

1. **Business Understanding** — four commercial questions → four models.
2. **Data Understanding** — `notebooks/02_data_understanding.ipynb`, Pandera schemas.
3. **Data Preparation** — `src/features/build.py`.
4. **Modeling** — `src/models/{elasticity,churn,segmentation,recommender}.py`.
5. **Evaluation** — MLflow metrics + `src/experiments/ab_test.py`.
6. **Deployment** — `pipelines/train.py`, MLflow registry, drift monitoring.

## Project layout

```
src/
  data/          synthetic generator, Pandera schemas, loader
  features/      elasticity, churn, segmentation, interaction builders
  models/        elasticity (statsmodels), churn (LightGBM), segmentation (KMeans/DBSCAN), recommender (implicit ALS)
  experiments/   A/B test simulation (Welch, two-proportion)
  monitoring/    Evidently drift report
pipelines/       end-to-end training pipeline with MLflow
tests/           pytest for data, features, models, A/B
```

## Metrics tracked (last run)

See `data/processed/run_summary.json` and the MLflow UI. The pipeline logs:

- `elasticity_r2` — fit quality of the log-log demand model
- `churn_auc`, `churn_pr_auc`, `churn_brier` — calibrated churn classifier
- `seg_k`, `seg_silhouette`, `seg_davies_bouldin` — KMeans quality
- `recall_at_10` — recommender hit rate
- `ab_p_value`, `ab_uplift_pct` — experiment significance

## Next steps

- Wire DVC (`dvc init`, `dvc add data/raw`) for dataset versioning
- Add MLflow Model Registry stage transitions (`Staging` → `Production`)
- Schedule `src/monitoring/drift.py` as a daily job
- Expand notebooks 01–08 for the full CRISP-DM narrative
