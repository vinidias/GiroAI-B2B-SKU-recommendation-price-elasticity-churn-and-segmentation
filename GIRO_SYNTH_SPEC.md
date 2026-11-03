# GIRO-SYNTH - Specification

**A synthetic B2B food-distribution dataset built for predictive modelling.**

Version 0.1.0 - draft for review

---

## 1. Why this dataset exists

The [Atacado Atlas](https://www.kaggle.com/datasets/vongrossi/atacado-atlas-synthetic-wholesale-distribution)
is excellent for **data engineering** (22 related tables, FEFO, routes, credit,
finance) but it was not built for **predictive modelling**. Measured on it:

| Model | Atlas metric | Why |
|---|---|---|
| Price elasticity (OLS) | R² = 0.004 | quantity is generated independently of price |
| Churn (LightGBM) | AUC = 0.537 | all 480 customers buy in all 9 quarters - no churners |
| SKU recommendation (ALS) | recall@10 = 0.208 | affinity is popularity-driven, no latent structure |

GIRO-SYNTH keeps the operational realism (same 12 categories, same 6 customer
segments, 5 price tables) but **injects the behavioural signals** a data
scientist needs to demonstrate: price response, churn dynamics and latent
customer-SKU affinity. Every signal is generated from a known ground truth so
model recovery can be measured.

---

## 2. Design principles

1. **Known ground truth.** Every injected signal has documented parameters
   (elasticities, churn hazards, latent factors) so a model can be scored
   against the truth, not just against held-out data.
2. **Operational realism preserved.** 12 categories, 6 segments, 5 price
   tables, BRL, FEFO lots, routes - same domain vocabulary as the Atlas.
3. **Controlled anomalies.** Like the Atlas, defects are injected on purpose
   and catalogued in `manifesto.csv` with expected clean counts.
4. **Reproducible.** Single seed, versioned, SHA-256 per file.
5. **Model-agnostic.** Plain CSVs + a schema.json; no framework lock-in.

---

## 3. Scale (v0.1.0)

| Entity | Count | Note |
|---|---|---|
| Period | 730 days (2024-09-01 to 2026-08-31) | same window as Atlas |
| SKUs | 144 | 12 per category x 12 categories |
| Customers | 480 | 80 per segment x 6 segments |
| Orders | ~55,000 | ~75 per customer |
| Line items | ~150,000 | 1-8 SKUs per order |
| Price tables | 5 | factors 0.88 - 1.12 over list price |
| Commodity series | 6 | CEPEA monthly means, 24 months |
| Ground-truth files | 4 | elasticity, churn hazards, latent factors, propensity |

---

## 4. Tables

### 4.1 Operational (mirror Atlas where possible)

```
clientes.csv          cliente_id, razao_social, segmento, uf, tabela_preco_id,
                      data_cadastro, canal, prazo_pagamento_dias, limite_credito_brl
produtos.csv          produto_id, descricao, categoria (12), familia,
                      unidade_venda, custo_referencia_brl, preco_lista_brl,
                      elasticidade_verdadeira (float, per SKU)
tabelas_preco.csv     tabela_preco_id, nome, segmento, fator_sobre_lista
precos_produto.csv    preco_produto_id, tabela_preco_id, produto_id, preco_brl,
                      vigencia_inicio, vigencia_fim
vendedores.csv        vendedor_id, nome, regiao, comissao_pct, meta_mensal_brl
pedidos.csv           pedido_id, cliente_id, vendedor_id, data_pedido,
                      status_pedido, canal_pedido, tabela_preco_aplicada,
                      valor_mercadoria_total_brl
itens_pedido.csv      item_pedido_id, pedido_id, produto_id, quantidade_solicitada,
                      quantidade_expedida, preco_unitario_brl, desconto_pct,
                      valor_mercadoria_brl
```

### 4.2 Behavioural ground truth (new - the whole point)

```
gt_elasticity.csv     categoria, epsilon (true elasticity),
                      intercept, price_range_min, price_range_max
gt_churn.csv          cliente_id, churn_date (or blank), churn_reason
                      (price | service | competitor | inactive),
                      hazard_base, hazard_multiplier
gt_affinity.csv       cliente_id, produto_id, affinity (float, 0-1),
                      latent_dim_1, latent_dim_2 (for visualisation)
gt_propensity.csv     cliente_id, base_order_rate (orders/month),
                      seasonality_amplitude, trend_slope
```

### 4.3 Quality artefacts

```
manifesto.csv         arquivo, linhas_raw, linhas_silver_esperadas, sha256
schema.json           column types + granularity per table
dicionario.csv        table, column, type, primary_key, description
CHECKSUMS.sha256      checksum of every CSV
```

---

## 5. Generative model

### 5.1 Price elasticity (the core fix)

For each category c, a true elasticity `epsilon_c` is drawn from:

    epsilon_c ~ Uniform(0.8, 2.0)      (inelastic -> elastic)

For each SKU, `elasticidade_verdadeira` = `epsilon_c` x lognormal(0, 0.15)
so SKUs within a category differ slightly.

**Demand generation.** For customer i, SKU j, month t:

    base_qty_ij    = propensity_i * affinity_ij * sku_popularity_j
    price_ratio    = price_ijt / list_price_j
    qty_ijt        ~ Poisson( base_qty_ij * price_ratio ** (-epsilon_j) * seasonality_t )

The `price_ratio ** (-epsilon_j)` term is the injected signal. Because price
varies across the 5 tables (factor 0.88-1.12) *and* across time (monthly list
price updates), an OLS on `log(qty) ~ log(price) + C(category)` must recover
`-epsilon_c` within sampling error.

**Acceptance test:** estimated elasticity per category within ±0.15 of truth
on 80% of categories, R² > 0.25.

### 5.2 Churn

Each customer has a **hazard** that grows over time:

    lambda_i(t) = hazard_base_i * exp(beta_drop * revenue_drop_i(t)
                                     + beta_late * late_payments_i(t)
                                     + beta_ticket * ticket_decline_i(t))

At each month, draw `churn ~ Bernoulli(1 - exp(-lambda_i(t)))`. Once churned,
the customer stops ordering (recency grows without bound).

**Correlated leading indicators** (so a model can actually predict):
- revenue in the trailing 3 months drops before churn
- payment delays increase in the 2 months before churn
- order frequency drops in the 3 months before churn
- SKU variety narrows in the 2 months before churn

**Acceptance test:** churn rate 12-18%, LightGBM AUC > 0.75 using only
pre-churn features.

### 5.3 Latent affinity (recommendation)

Each customer and SKU gets a **d-dimensional latent vector** (d = 8):

    u_i ~ Normal(0, 1)^d
    v_j ~ Normal(0, 1)^d
    affinity_ij = sigmoid(u_i . v_j / sqrt(d))

Affinity modulates which SKUs a customer buys. Two customers who bought the
same SKU will tend to buy the same *other* SKUs - the structure ALS needs.

**Acceptance test:** ALS recall@10 > 0.35 (vs 0.208 on Atlas).

### 5.4 Seasonality and trend

    seasonality_t = 1 + 0.15 * sin(2*pi*(t/12) + phase_category)
    trend_t       = 1 + trend_slope_i * (t / 24)

Applied to order propensity so monthly volumes are realistic.

---

## 6. Controlled anomalies (catalogued in manifesto)

| Anomaly | Rows | Purpose |
|---|---|---|
| Exact duplicate orders | 25 | dedup test |
| Duplicate order with non-existent customer | 1 | referential integrity |
| Textual junk in numeric column ("INVALIDO") | 12 | coercion test |
| Capitalization drift (Faturado / FATURADO) | 4 | normalization test |
| Discount as percent not fraction | all itens | unit test |
| Duplicate line items | 29 | dedup test |

Expected clean counts are written to `manifesto.csv` so any loader can verify.

---

## 7. File layout

```
giro-synth/
  README.md
  LICENSE.md                 (CC BY 4.0)
  schema.json
  dicionario.csv
  manifesto.csv
  CHECKSUMS.sha256
  data/
    operational/             clientes, produtos, pedidos, itens, precos, ...
    ground_truth/            gt_elasticity, gt_churn, gt_affinity, gt_propensity
    macro/                   cepea_monthly.csv (from agrobr)
  generator/
    generate.py              entry point (--seed, --customers, --days)
    config.yaml              all tunable parameters
  notebooks/
    01_verify_signals.ipynb  proves the injected signals are recoverable
```

---

## 8. Validation harness (shipped with the dataset)

The repo includes a notebook and a script that run the Giro AI pipeline on
GIRO-SYNTH and check:

| Model | Acceptance |
|---|---|
| Elasticity | R² > 0.25, elasticity within ±0.15 for 80% of categories |
| Churn | AUC > 0.75, PR-AUC > 0.40 |
| Recommendation | recall@10 > 0.35 |
| Segmentation | silhouette > 0.45 |

If any fails, the dataset is not ready to publish.

---

## 9. Kaggle dataset card (draft)

**Title:** GIRO-SYNTH - B2B Food Distribution with Known Behavioural Signals

**Subtitle:** 144 SKUs, 480 customers, 730 days - price elasticity, churn and
latent affinity generated from documented ground truth.

**Why:** existing synthetic B2B datasets are built for data engineering. This
one is built for **predictive modelling**: every signal a model is supposed to
learn is injected with a known parameter, so results can be scored against the
truth.

**Contents:** 22 operational CSVs + 4 ground-truth CSVs + validation notebook.

**License:** CC BY 4.0 (data), MIT (generator).

---

## 10. Open questions for review

1. **Scale.** 480 customers is enough for a demo but small for deep models.
   Bump to 2,000 customers / 300 SKUs / 200k orders?
2. **Categories.** Keep the 12 Atlas categories for continuity, or redesign?
3. **Ground truth exposure.** Ship `gt_*.csv` in the public dataset, or keep
   them in a separate private split so users cannot cheat?
4. **Macro coupling.** Make commodity prices *drive* costs and therefore prices,
   or keep them as an independent covariate (as now)?
