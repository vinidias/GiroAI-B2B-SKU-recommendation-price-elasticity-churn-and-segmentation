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

