"""Feature builders for the four business models (real Atacado Atlas schema).

Source tables (after validate.load_and_validate_all):
  pedidos        pedido_id, cliente_id, vendedor_id, data_pedido, status_pedido,
                 canal_pedido, tabela_preco_aplicada, valor_mercadoria_total_brl
  itens_pedido   item_pedido_id, pedido_id, produto_id, quantidade_solicitada,
                 quantidade_expedida, preco_unitario_brl, desconto_pct,
                 valor_mercadoria_brl
  produtos       produto_id, categoria (12), familia, custo_referencia_brl,
                 preco_lista_brl
  clientes       cliente_id, segmento (6), uf, vendedor_id, tabela_preco_id

Only FATURADO orders are treated as realized sales; CANCELADO, SEM_EXPEDICAO and
BLOQUEADO_CREDITO are excluded from revenue-based features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VALID_STATUS = "FATURADO"


def _order_totals(pedidos: pd.DataFrame, itens: pd.DataFrame) -> pd.DataFrame:
    agg = itens.groupby("pedido_id").agg(
        n_skus=("produto_id", "nunique"),
        n_itens=("quantidade_expedida", "sum"),
        receita=("valor_mercadoria_brl", "sum"),
    ).reset_index()
    out = pedidos.merge(agg, on="pedido_id", how="left")
    return out.fillna({"n_skus": 0, "n_itens": 0, "receita": 0.0})


def build_elasticity_features(pedidos, itens, produtos, clientes,
                              cepea_daily=None, usda_df=None,
                              with_macro: bool = True) -> pd.DataFrame:
    """Price-quantity panel for the log-log elasticity regression.

    Effective price = preco_unitario_brl * (1 - desconto_pct). The Atlas also
    exposes tabela_preco_aplicada (5 price tables with a factor over list price),
    kept as a categorical control via the client's segment.
    """
    df = itens.merge(
        pedidos[["pedido_id", "cliente_id", "data_pedido", "status_pedido",
                 "canal_pedido", "tabela_preco_aplicada"]],
        on="pedido_id", how="left",
    )
    df = df[df["status_pedido"] == VALID_STATUS]
    df = df.merge(clientes[["cliente_id", "segmento", "uf"]], on="cliente_id", how="left")
    df = df.merge(
        produtos[["produto_id", "categoria", "custo_referencia_brl"]],
        on="produto_id", how="left",
    )
    df["preco_liquido"] = df["preco_unitario_brl"] * (1 - df["desconto_pct"])
    df["mes"] = pd.to_datetime(df["data_pedido"]).dt.month
    df["log_qty"] = np.log1p(df["quantidade_expedida"].clip(lower=0))
    df["log_preco"] = np.log(df["preco_liquido"].clip(lower=1e-6))
    df["log_custo"] = np.log(df["custo_referencia_brl"].clip(lower=1e-6))
    df["margem_unit"] = df["preco_liquido"] - df["custo_referencia_brl"]
    df = df[df["quantidade_expedida"] > 0]
    out = df[["pedido_id", "cliente_id", "produto_id", "categoria", "segmento",
              "uf", "canal_pedido", "data_pedido", "mes", "log_qty", "log_preco",
              "log_custo", "margem_unit", "valor_mercadoria_brl"]].dropna(
        subset=["log_preco", "log_qty"])

    if with_macro:
        try:
            from src.data.macro import attach_macro_features
            out = attach_macro_features(out, cepea_daily=cepea_daily, usda_df=usda_df)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Macro features skipped: %s", exc)
            out["preco_commodity"] = np.nan
            out["log_preco_commodity"] = np.nan
            out["producao_usda"] = np.nan
            out["log_producao_usda"] = np.nan
    return out


def build_churn_features(clientes, pedidos, itens,
                         reference_date: pd.Timestamp | None = None,
                         drop_threshold: float = 0.5) -> pd.DataFrame:
    """Behavioural features + binary churn label defined by revenue decline.

    The Atlas has no churn label and, in fact, no churners: all 480 customers
    buy in all 9 quarters and max recency is ~77 days. A recency-based label
    produces a target that only recency itself can explain (AUC 0.46 without
    leakage). We therefore define churn behaviourally:

        churn = 1 if revenue in the last 180 days < drop_threshold * revenue
                in the 180 days before that (default drop > 50%).

    Label: churn (0/1).
    """
    pedidos = pedidos.copy()
    pedidos["data_pedido"] = pd.to_datetime(pedidos["data_pedido"])
    ref = reference_date or (pedidos["data_pedido"].max() + pd.Timedelta(days=1))
    totals = _order_totals(pedidos, itens)
    totals = totals[totals["status_pedido"] == VALID_STATUS]

    g = totals.groupby("cliente_id")
    feats = pd.DataFrame({
        "recencia_dias": (ref - g["data_pedido"].max()).dt.days,
        "frequencia": g["pedido_id"].nunique(),
        "ticket_medio": g["receita"].mean(),
        "receita_total": g["receita"].sum(),
        "skus_distintos": g["n_skus"].sum(),
    }).reset_index()

    half = pd.Timedelta(days=180)
    last6 = (totals[totals["data_pedido"] > ref - half]
             .groupby("cliente_id")["receita"].sum().rename("receita_6m"))
    prev6 = (totals[(totals["data_pedido"] <= ref - half)
                    & (totals["data_pedido"] > ref - 2 * half)]
             .groupby("cliente_id")["receita"].sum().rename("receita_6m_anterior"))
    revenue = pd.concat([last6, prev6], axis=1).fillna(0).reset_index()

    out = clientes.merge(feats, on="cliente_id", how="left")
    out = out.merge(revenue, on="cliente_id", how="left")
    cols = ["recencia_dias", "frequencia", "ticket_medio", "receita_total",
            "skus_distintos", "receita_6m", "receita_6m_anterior"]
    out[cols] = out[cols].fillna(0)

    out["variacao_receita_pct"] = (
        (out["receita_6m"] - out["receita_6m_anterior"])
        / out["receita_6m_anterior"].replace(0, np.nan)
    ).fillna(0)
    out["churn"] = (
        (out["receita_6m_anterior"] > 0)
        & (out["receita_6m"] < drop_threshold * out["receita_6m_anterior"])
    ).astype(int)
    out.attrs["churn_drop_threshold"] = drop_threshold
    return out


def build_segmentation_features(clientes, pedidos, itens) -> pd.DataFrame:
    """RFM matrix for KMeans / DBSCAN clustering."""
    pedidos = pedidos.copy()
    pedidos["data_pedido"] = pd.to_datetime(pedidos["data_pedido"])
    ref = pedidos["data_pedido"].max() + pd.Timedelta(days=1)
    totals = _order_totals(pedidos, itens)
    totals = totals[totals["status_pedido"] == VALID_STATUS]
    g = totals.groupby("cliente_id")
    rfm = pd.DataFrame({
        "recencia": (ref - g["data_pedido"].max()).dt.days,
        "frequencia": g["pedido_id"].nunique(),
        "monetario": g["receita"].sum(),
        "n_skus": g["n_skus"].sum(),
    }).reset_index()
    return clientes.merge(rfm, on="cliente_id", how="left").fillna(0)


def build_interactions(pedidos, itens) -> pd.DataFrame:
    """Implicit feedback matrix: (cliente_id, produto_id) -> quantity bought."""
    df = itens.merge(pedidos[["pedido_id", "cliente_id", "status_pedido"]], on="pedido_id")
    df = df[(df["status_pedido"] == VALID_STATUS) & (df["quantidade_expedida"] > 0)]
    return df.groupby(["cliente_id", "produto_id"]).agg(
        n=("quantidade_expedida", "sum"),
        spend=("valor_mercadoria_brl", "sum"),
    ).reset_index()
