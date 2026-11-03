# Giro AI

> End-to-end data science platform for B2B food distribution: SKU recommendation, price elasticity, churn prediction and customer segmentation - built with CRISP-DM, MLflow and A/B testing.

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
.venv\Scripts\activate
pip install -e ".[dev]"

python -m src.data.synthetic
python -m pipelines.train
mlflow ui --backend-store-uri ./mlruns
```

## Using the real Atacado Atlas dataset

Drop these CSVs into `data/raw/`:

```
clientes.csv  produtos.csv  pedidos.csv
itens_pedido.csv  precos_produto.csv  vendedores.csv
```

The loader validates them against Pandera schemas and the pipeline runs unchanged.

## CRISP-DM

1. Business Understanding - four commercial questions -> four models
2. Data Understanding - Pandera schemas + EDA notebooks
3. Data Preparation - `src/features/build.py`
4. Modeling - `src/models/`
5. Evaluation - MLflow metrics + A/B tests
6. Deployment - `pipelines/train.py` + drift monitoring

## Metrics tracked

- `elasticity_r2`, `churn_auc`, `churn_pr_auc`, `churn_brier`
- `seg_k`, `seg_silhouette`, `seg_davies_bouldin`
- `recall_at_10`, `ab_p_value`, `ab_uplift_pct`

 
﻿

## Macro features (agrobr / CEPEA)

The elasticity model is enriched with real Brazilian commodity prices fetched via
[**agrobr**](https://github.com/bruno-portfolio/agrobr) (CEPEA/ESALQ daily spot):

| Category | CEPEA commodity |
|---|---|
| Bebidas | cafe |
| Mercearia | milho |
| Limpeza | algodao |
| Perecíveis | boi_gordo |
| Congelados | frango_congelado |
| Higiene | soja |

- src/data/macro.py fetches the 6 indicators (5,742 daily observations), caches to data/interim/cepea_daily.parquet, aggregates monthly, and joins on ano_mes + commodity.
- src/models/elasticity.py adds log_preco_commodity as a covariate when available.
- Fails gracefully: if the network is down, the model runs without the macro term.
- 3 hermetic tests in tests/test_macro.py (no network needed).

**License note**: CEPEA data is CC BY-NC 4.0 - attribution required, commercial use needs CEPEA authorization.

### USDA PSD (optional, requires API key)

`src/data/macro.py` also fetches annual USDA Production, Supply and Distribution
balances for 4 commodities (cafe, milho, algodao, soja) across 5 market years
(295 rows cached). Requires `AGROBR_USDA_API_KEY` in `.env` (free at
https://api.data.gov/signup/).

**Modeling note**: `log_producao_usda` is kept in the feature frame for EDA but
is **not** used by the OLS. Being constant within each category x year, it is
collinear with the category dummies and would make the design matrix
rank-deficient (verified: rank 10 of 14 columns). The elasticity model uses only
CEPEA's `log_preco_commodity`, which varies within category x month (rank 14 of 14).

