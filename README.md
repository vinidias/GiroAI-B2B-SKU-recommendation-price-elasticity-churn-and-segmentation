# Giro AI

> End-to-end data science platform for B2B food distribution: SKU recommendation, price elasticity, churn prediction and customer segmentation - built with CRISP-DM, MLflow and A/B testing.

**Dataset:** [Atacado Atlas](https://www.kaggle.com/datasets/vongrossi/atacado-atlas-synthetic-wholesale-distribution) - 22 CSVs, 730 days, 144 SKUs, 480 wholesale customers, 52k orders, 147k line items.

## Business impact

| Question | Model | Gain |
|---|---|---|
| What should we recommend to this customer? | Implicit ALS collaborative filtering | Higher AOV, cross-sell revenue |
| How will demand react if we change the price? | Log-log OLS per category | Margin-aware pricing |
| Which customers are about to reduce spend? | Calibrated LightGBM | Proactive retention |
| Which customers behave alike? | KMeans / DBSCAN | Targeted CRM and territory design |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

python -m pipelines.train
mlflow ui --backend-store-uri sqlite:///mlflow.db
pytest -v
```

## Data quality (Atlas anomalies handled)

The Atlas intentionally ships controlled data-quality defects. `src/data/validate.py`
cleans them and logs every removal so the pipeline is auditable:

| Anomaly | Raw | Clean | Handling |
|---|---|---|---|
| Duplicate order rows | 52,422 | 52,397 | exact + business-key dedup (manifesto) |
| Order pointing at non-existent customer C9999 | +1 row | dropped | business-key dedup keeps first |
| Junk values in numeric columns ("INVALIDO") | 12 rows | dropped | coerce + drop on critical cols |
| Duplicate line items | 147,082 | 147,042 | same as above |
| Capitalization drift (Faturado / FATURADO) | 4 rows | normalized | UPPER + strip on categorical cols |
| Discount stored as percent (0-8) | all rows | /100 | normalized to fraction |

Every step is covered by `tests/test_data_validation.py`.

## Results on the real Atlas data

| Model | Metric | Value | Reading |
|---|---|---|---|
| Recommender (ALS) | recall@10 | **0.225** | solid for 144 SKUs / 480 users |
| Segmentation (KMeans) | silhouette | **0.413** | acceptable, k=2 |
| Churn (LightGBM) | ROC-AUC | **0.537** | weak - see note below |
| Elasticity (OLS) | R² | **0.004** | weak - see note below |
| A/B simulation | uplift / p | 15.6% / ~0 | synthetic uplift, validates the harness |

### Honest notes on the weak signals

**Elasticity R² = 0.004.** The Atlas assigns prices through 5 price tables
(`fator_sobre_lista` from 0.88 to 1.12) but does not simulate demand response to
those prices. Quantity is generated independently of price, so there is no
elasticity signal to recover. The model is correctly specified and would work on
data with real price variation; the limitation is the dataset, not the code.

**Churn AUC = 0.537.** All 480 Atlas customers are ATIVO and buy in all 9
quarters, so the dataset contains no churners. We define churn behaviourally
(revenue in last 180 days < 50% of the prior 180 days -> 21 customers, 4.4%).
RFM features barely predict that decline, which is the honest result: the Atlas
was built for operational/analytics study, not retention modelling. A
recency-based label would give AUC 0.47 and a leaky AUC 1.00 if recency were
left in the features; both were rejected.

The recommender and segmentation results are the ones that hold up on this data.

## CRISP-DM

1. **Business Understanding** - four commercial questions -> four models
2. **Data Understanding** - 22-table Atlas, Pandera schemas, anomaly catalogue
3. **Data Preparation** - `src/features/build.py` (RFM, price panel, interactions)
4. **Modeling** - `src/models/` (statsmodels, LightGBM, KMeans/DBSCAN, implicit ALS)
5. **Evaluation** - MLflow metrics + `src/experiments/ab_test.py`
6. **Deployment** - `pipelines/train.py` + `src/monitoring/drift.py` (Evidently)

## Macro features (agrobr / CEPEA + USDA)

The elasticity model is enriched with real Brazilian commodity prices fetched via
[**agrobr**](https://github.com/bruno-portfolio/agrobr):

| Atlas category | CEPEA commodity |
|---|---|
| BEBIDAS | cafe |
| MERCEARIA / PADARIA / SNACKS / PET | milho |
| LIMPEZA / HIGIENE / DESCARTAVEIS | algodao |
| FRIOS / LATICINIOS | boi_gordo |
| CONGELADOS | frango_congelado |
| HORTIFRUTI | soja |

- **CEPEA** - 5,742 daily observations (6 commodities), cached, aggregated monthly, used in the OLS as `log_preco_commodity`.
- **USDA PSD** - 295 annual observations (4 commodities x 5 years), requires `AGROBR_USDA_API_KEY`. Kept in the feature frame for EDA but **not** in the OLS: production is constant within category x year and collinear with the category dummies (rank 10 of 14).
- Fails gracefully: without network/key the model runs with NaN macro columns.

**License note**: CEPEA data is CC BY-NC 4.0 - attribution required, commercial use needs CEPEA authorization.

## Layout

```
src/
  config.py
  data/          validate.py (Atlas schemas + cleaning) · macro.py (CEPEA/USDA) · synthetic.py (fallback)
  features/      build.py (elasticity, churn, segmentation, interactions)
  models/        elasticity · churn · segmentation · recommender · evaluate
  experiments/   ab_test.py (Welch + two-proportion)
  monitoring/    drift.py (Evidently)
pipelines/       train.py (MLflow end-to-end)
tests/           24 tests across data, features, models, A/B, macro
```

## Metrics tracked

- `elasticity_r2`, `elasticity_has_cepea`, `elasticity_usda_feature_rows`
- `churn_auc`, `churn_pr_auc`, `churn_brier`
- `seg_k`, `seg_silhouette`, `seg_davies_bouldin`
- `recall_at_10`, `ab_p_value`, `ab_uplift_pct`
 
﻿

## Atlas vs GIRO-SYNTH: what a known ground truth changes

The same Giro AI pipeline, run on both datasets, shows the difference between
a dataset built for data engineering and one built for modelling:

| Model | Atacado Atlas | GIRO-SYNTH | Why |
|---|---|---|---|
| Price elasticity (R²) | 0.004 | **0.700** | Atlas generates quantity independently of price |
| Churn (ROC-AUC) | 0.537 | **0.743** | Atlas has zero churners; GIRO-SYNTH has a real hazard |
| Recommendation (recall@10) | 0.208 | **0.244** | GIRO-SYNTH has latent affinity, not just popularity |
| Segmentation (silhouette) | 0.413 | **0.559** | richer behavioural spread |

Two fixes were required in the Giro AI code to make this comparison fair, and
they are now the default:

- **Elasticity** uses a within-SKU x month fixed-effects OLS. The pooled
  specification conflated price-table assignment, monthly list changes and
  per-order promotions; only the last is demand response.
- **Recall@10** uses leave-one-out over 500 sampled users. The previous "last
  item per user" proxy held out each user's smallest purchase and understated
  recall by ~16x.

**GIRO-SYNTH** (generator + dataset card): https://github.com/vinidias/GiroAI-SYNTH

