"""Feature builders for the four business models."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _order_totals(pedidos: pd.DataFrame, itens: pd.DataFrame) -> pd.DataFrame:
    agg = itens.groupby("pedido_id").agg(
        n_skus=("produto_id", "nunique"),
        n_itens=("quantidade", "sum"),
        receita=("valor_total", "sum"),
    ).reset_index()
    return pedidos.merge(agg, on="pedido_id", how="left").fillna({"n_skus": 0, "n_itens": 0, "receita": 0.0})


def build_elasticity_features(pedidos, itens, produtos, clientes,
                                cepea_daily=None, with_macro: bool = True) -> pd.DataFrame:
    df = itens.merge(pedidos[["pedido_id", "cliente_id", "data_pedido"]], on="pedido_id", how="left")
    df = df.merge(clientes[["cliente_id", "canal", "regiao"]], on="cliente_id", how="left")
    df = df.merge(produtos[["produto_id", "categoria", "custo_unitario"]], on="produto_id", how="left")
    df["preco_liquido"] = df["preco_unitario"] * (1 - df["desconto_pct"])
    df["mes"] = pd.to_datetime(df["data_pedido"]).dt.month
    df["log_qty"] = np.log1p(df["quantidade"])
    df["log_preco"] = np.log(df["preco_liquido"].clip(lower=1e-6))
    df["log_custo"] = np.log(df["custo_unitario"].clip(lower=1e-6))
    df["margem_unit"] = df["preco_liquido"] - df["custo_unitario"]
    out = df[["pedido_id", "cliente_id", "produto_id", "categoria", "canal", "regiao",
              "data_pedido", "mes", "log_qty", "log_preco", "log_custo",
              "margem_unit", "valor_total"]].dropna(subset=["log_preco"])

    if with_macro:
        try:
            from src.data.macro import attach_macro_features
            out = attach_macro_features(out, cepea_daily=cepea_daily)
        except Exception as exc:  # keep pipeline resilient
            import logging
            logging.getLogger(__name__).warning("Macro features skipped: %s", exc)
            out["preco_commodity"] = np.nan
            out["log_preco_commodity"] = np.nan
    return out


def build_churn_features(clientes, pedidos, itens, reference_date=None) -> pd.DataFrame:
    pedidos = pedidos.copy()
    pedidos["data_pedido"] = pd.to_datetime(pedidos["data_pedido"])
    ref = reference_date or (pedidos["data_pedido"].max() + pd.Timedelta(days=1))
    totals = _order_totals(pedidos, itens)
    totals = totals[totals["status"] == "entregue"]
    g = totals.groupby("cliente_id")
    feats = pd.DataFrame({
        "recencia_dias": (ref - g["data_pedido"].max()).dt.days,
        "frequencia": g["pedido_id"].nunique(),
        "ticket_medio": g["receita"].mean(),
        "receita_total": g["receita"].sum(),
        "skus_distintos": g["n_skus"].sum(),
        "ultimo_pedido_dias": (ref - g["data_pedido"].max()).dt.days,
    }).reset_index()
    out = clientes.merge(feats, on="cliente_id", how="left")
    cols = ["recencia_dias", "frequencia", "ticket_medio", "receita_total", "skus_distintos", "ultimo_pedido_dias"]
    out[cols] = out[cols].fillna(0)
    out["churn"] = out["churn"].fillna(0).astype(int) if "churn" in out.columns else (out["recencia_dias"] > 90).astype(int)
    return out


def build_segmentation_features(clientes, pedidos, itens) -> pd.DataFrame:
    pedidos = pedidos.copy()
    pedidos["data_pedido"] = pd.to_datetime(pedidos["data_pedido"])
    ref = pedidos["data_pedido"].max() + pd.Timedelta(days=1)
    totals = _order_totals(pedidos, itens)
    totals = totals[totals["status"] == "entregue"]
    g = totals.groupby("cliente_id")
    rfm = pd.DataFrame({
        "recencia": (ref - g["data_pedido"].max()).dt.days,
        "frequencia": g["pedido_id"].nunique(),
        "monetario": g["receita"].sum(),
        "n_skus": g["n_skus"].sum(),
    }).reset_index()
    return clientes.merge(rfm, on="cliente_id", how="left").fillna(0)


def build_interactions(pedidos, itens) -> pd.DataFrame:
    df = itens.merge(pedidos[["pedido_id", "cliente_id", "status"]], on="pedido_id")
    df = df[df["status"] == "entregue"]
    return df.groupby(["cliente_id", "produto_id"]).agg(
        n=("quantidade", "sum"), spend=("valor_total", "sum")).reset_index()

