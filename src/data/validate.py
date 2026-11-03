"""Pandera schemas + loader for the real Atacado Atlas tables.

Column names and file layout mirror the Atlas release (22 CSVs, 730 days,
144 SKUs, 480 customers).

The Atlas intentionally ships **14 controlled data-quality anomalies**. They are
documented in README.md and manifesto.csv, and this loader cleans or flags each
one so the pipeline is auditable:

  1. exact duplicate rows (pedidos 52422 -> 52397, itens_pedido 147082 -> 147053)
  2. a duplicate order pointing at a non-existent customer (C9999)
  3. textual junk in numeric columns (e.g. "INVALIDO" in preco_unitario_brl)
  4. duplicated item_pedido_id rows
  5. capitalization drift (Faturado / FATURADO, Mercearia / MERCEARIA)
  6. discount stored as percent (0-8) instead of fraction
  7. box-conversion price inflation: CX SKUs listed at 100-450x cost in both
     produtos.preco_lista_brl and itens_pedido.preco_unitario_brl
     (25,555 of 147,042 items, 17%) - R$211M of phantom revenue
  8. decimal comma in price (12 items, e.g. "11,09")
  9. discount shown but not applied: 73,415 items have valor_mercadoria =
     qty * preco while desconto_pct > 0 (the other 9,099 are correct)
 10. negative quantidade_solicitada (1 item: IT0000004)
 11. mixed date formats: 12 pedidos in DD/MM/YYYY, the rest in ISO
 12. orphan FK: itens_pedido.produto_id = P999 (1 item)
 13. optional nulls that are legitimate: validade_dias / temperatura_max_c
     blank for the 84 non-perishable SKUs; frete_rateado_brl blank for the
     5,001 non-invoiced orders
 14. 12 pedidos with an unparseable data_pedido (same rows as anomaly 11)

The loader reads every column as string first (so pandas never silently turns
junk into NaN), drops duplicates, coerces types, drops rows that fail on
critical columns, and normalises the anomalies above.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

from src.config import DATA_DIR

log = logging.getLogger(__name__)

RAW_FILES = {
    "clientes": "clientes.csv",
    "produtos": "produtos.csv",
    "pedidos": "pedidos.csv",
    "itens_pedido": "itens_pedido.csv",
    "precos_produto": "precos_produto.csv",
    "tabelas_preco": "tabelas_preco.csv",
    "vendedores": "vendedores.csv",
}

PRIMARY_KEYS = {
    "clientes": "cliente_id",
    "produtos": "produto_id",
    "pedidos": "pedido_id",
    "itens_pedido": "item_pedido_id",
    "precos_produto": "preco_produto_id",
    "tabelas_preco": "tabela_preco_id",
    "vendedores": "vendedor_id",
}

DATE_COLS = {
    "pedidos": ["data_pedido", "data_programacao"],
    "clientes": ["data_cadastro"],
    "precos_produto": ["vigencia_inicio", "vigencia_fim"],
    "tabelas_preco": ["vigencia_inicio", "vigencia_fim"],
    "vendedores": ["data_admissao"],
}

NUMERIC_COLS = {
    "pedidos": ["valor_mercadoria_total_brl", "peso_total_solicitado_kg",
                "frete_rateado_brl"],
    "itens_pedido": ["quantidade_solicitada", "quantidade_expedida",
                     "preco_unitario_brl", "valor_mercadoria_brl"],
    "produtos": ["custo_referencia_brl", "preco_lista_brl"],
    "precos_produto": ["preco_brl"],
    "tabelas_preco": ["fator_sobre_lista"],
}

CATEGORICAL_COLS = {
    "pedidos": ["status_pedido", "canal_pedido", "tabela_preco_aplicada", "moeda"],
    "clientes": ["segmento", "uf", "situacao"],
    "produtos": ["categoria", "familia", "unidade_venda", "unidade_estoque", "situacao"],
    "vendedores": ["regiao", "situacao"],
    "tabelas_preco": ["segmento"],
}

CRITICAL_COLS = {
    "pedidos": ["data_pedido", "valor_mercadoria_total_brl"],
    "itens_pedido": ["pedido_id", "produto_id", "quantidade_solicitada",
                     "preco_unitario_brl"],
    "produtos": ["custo_referencia_brl", "preco_lista_brl"],
    "precos_produto": ["preco_brl", "vigencia_inicio"],
    "tabelas_preco": ["fator_sobre_lista"],
}

MARKUP_SANITY_MAX = 50.0
DESCONTO_SANITY = 1.0  # desconto is a fraction after normalisation

SCHEMAS: dict[str, DataFrameSchema] = {
    "clientes": DataFrameSchema({
        "cliente_id": Column(str, unique=True),
        "razao_social": Column(str),
        "segmento": Column(str),
        "uf": Column(str),
        "vendedor_id": Column(str),
        "tabela_preco_id": Column(str),
    }, coerce=True, strict=False),
    "produtos": DataFrameSchema({
        "produto_id": Column(str, unique=True),
        "descricao": Column(str),
        "categoria": Column(str),
        "familia": Column(str),
        "custo_referencia_brl": Column(float, Check.ge(0)),
        "preco_lista_brl": Column(float, Check.ge(0)),
    }, coerce=True, strict=False),
    "pedidos": DataFrameSchema({
        "pedido_id": Column(str, unique=True),
        "cliente_id": Column(str),
        "vendedor_id": Column(str),
        "data_pedido": Column("datetime64[ns]"),
        "status_pedido": Column(str, Check.isin(
            ["FATURADO", "CANCELADO", "SEM_EXPEDICAO", "BLOQUEADO_CREDITO"])),
        "canal_pedido": Column(str),
        "tabela_preco_aplicada": Column(str),
        "valor_mercadoria_total_brl": Column(float, Check.ge(0)),
    }, coerce=True, strict=False),
    "itens_pedido": DataFrameSchema({
        "item_pedido_id": Column(str, unique=True),
        "pedido_id": Column(str),
        "produto_id": Column(str),
        "quantidade_solicitada": Column(int, Check.ge(0)),
        "quantidade_expedida": Column(int, Check.ge(0)),
        "preco_unitario_brl": Column(float, Check.gt(0)),
        "desconto_pct": Column(float, Check.in_range(0, 1)),
        "valor_mercadoria_brl": Column(float, Check.ge(0)),
    }, coerce=True, strict=False),
    "precos_produto": DataFrameSchema({
        "preco_produto_id": Column(str, unique=True),
        "tabela_preco_id": Column(str),
        "produto_id": Column(str),
        "preco_brl": Column(float, Check.gt(0)),
        "vigencia_inicio": Column("datetime64[ns]"),
    }, coerce=True, strict=False),
    "tabelas_preco": DataFrameSchema({
        "tabela_preco_id": Column(str, unique=True),
        "nome": Column(str),
        "fator_sobre_lista": Column(float, Check.gt(0)),
    }, coerce=True, strict=False),
    "vendedores": DataFrameSchema({
        "vendedor_id": Column(str, unique=True),
        "nome": Column(str),
        "regiao": Column(str),
    }, coerce=True, strict=False),
}


def _parse_number(s: pd.Series) -> pd.Series:
    """Parse numbers that may carry a decimal comma (anomaly 8)."""
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False),
                         errors="coerce")


def _coerce(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS.get(name, []):
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()
    for col in DATE_COLS.get(name, []):
        if col in df.columns:
            # Anomaly 11: 12 pedidos arrive as DD/MM/YYYY. Try ISO first, then
            # dayfirst, so both formats land in the same dtype.
            s = df[col]
            parsed = pd.to_datetime(s, errors="coerce", format="ISO8601")
            needs_dayfirst = parsed.isna() & s.notna() & (s != "")
            if needs_dayfirst.any():
                parsed.loc[needs_dayfirst] = pd.to_datetime(
                    s[needs_dayfirst], errors="coerce", dayfirst=True)
            df[col] = parsed
    for col in NUMERIC_COLS.get(name, []):
        if col in df.columns:
            df[col] = _parse_number(df[col])
    # Anomaly 6: discount stored as percent
    if name == "itens_pedido" and "desconto_pct" in df.columns:
        df["desconto_pct"] = _parse_number(df["desconto_pct"]) / 100.0
    return df


def load_raw(name: str, data_dir: Path | None = None,
             drop_duplicates: bool = True) -> pd.DataFrame:
    """Load one Atlas table, cleaning controlled anomalies."""
    data_dir = data_dir or DATA_DIR
    path = data_dir / RAW_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}")

    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    n_raw = len(raw)

    if drop_duplicates:
        raw = raw.drop_duplicates()
        key = PRIMARY_KEYS.get(name)
        if key and key in raw.columns:
            # anomaly 4b: one line item has an empty primary key; the manifesto
            # count (147082 -> 147053) only reconciles if it is dropped.
            n_before = len(raw)
            raw = raw[raw[key].astype(str).str.strip() != ""]
            if len(raw) < n_before:
                log.info("%s: dropped %d rows with empty primary key",
                         name, n_before - len(raw))
            raw = raw.drop_duplicates(subset=[key], keep="first")

    df = _coerce(raw, name)

    critical = CRITICAL_COLS.get(name, [])
    if critical:
        before = len(df)
        df = df.dropna(subset=[c for c in critical if c in df.columns])
        if len(df) < before:
            log.info("%s: dropped %d rows with invalid critical columns",
                     name, before - len(df))

    if len(df) < n_raw:
        log.info("%s: %d raw -> %d clean rows", name, n_raw, len(df))
    return df


def validate(name: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df if df is not None else load_raw(name)
    schema = SCHEMAS.get(name)
    return schema.validate(df) if schema is not None else df


def clean_prices(produtos: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Anomaly 7: normalise inflated list prices (box conversion applied twice)."""
    p = produtos.copy()
    p["_markup"] = p["preco_lista_brl"] / p["custo_referencia_brl"].replace(0, np.nan)
    sane = p[p["_markup"] <= MARKUP_SANITY_MAX]
    cat_markup = sane.groupby("categoria")["_markup"].median()
    bad = p["_markup"] > MARKUP_SANITY_MAX
    n_fixed = int(bad.sum())
    if n_fixed:
        p.loc[bad, "preco_lista_brl"] = (
            p.loc[bad, "custo_referencia_brl"] * p.loc[bad, "categoria"].map(cat_markup)
        ).round(2)
        log.info("produtos: %d SKUs with inflated box prices normalised", n_fixed)
    return p.drop(columns=["_markup"]), n_fixed


def clean_items(itens: pd.DataFrame, produtos: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean anomalies 7, 9, 10, 12 on the line-item table.

    - anomaly 7: inflated unit price (box conversion applied twice)
    - anomaly 9: desconto_pct shown but not applied to valor_mercadoria_brl.
      We recompute valor_mercadoria = qty * preco * (1 - desconto) so the
      discount is consistent with the displayed field.
    - anomaly 10: negative quantidade_solicitada -> clipped to 0
    - anomaly 12: orphan produto_id P999 -> row dropped
    """
    df = itens.merge(
        produtos[["produto_id", "categoria", "custo_referencia_brl"]],
        on="produto_id", how="left",
    )
    counts = {}

    # anomaly 12: orphan SKU
    orphan = df["categoria"].isna()
    counts["orphan_produto"] = int(orphan.sum())
    if counts["orphan_produto"]:
        df = df[~orphan].copy()
        log.info("itens_pedido: %d items with orphan produto_id dropped",
                 counts["orphan_produto"])

    # anomaly 10: negative quantity
    neg = df["quantidade_solicitada"] < 0
    counts["negative_qty"] = int(neg.sum())
    if counts["negative_qty"]:
        df.loc[neg, "quantidade_solicitada"] = 0
        log.info("itens_pedido: %d negative quantities clipped to 0", counts["negative_qty"])

    # anomaly 7: inflated price
    df["_markup"] = df["preco_unitario_brl"] / df["custo_referencia_brl"].replace(0, np.nan)
    sane = df[df["_markup"] <= MARKUP_SANITY_MAX]
    cat_markup = sane.groupby("categoria")["_markup"].median()
    inflated = df["_markup"] > MARKUP_SANITY_MAX
    counts["inflated_price"] = int(inflated.sum())
    if counts["inflated_price"]:
        df.loc[inflated, "preco_unitario_brl"] = (
            df.loc[inflated, "custo_referencia_brl"]
            * df.loc[inflated, "categoria"].map(cat_markup)
        ).round(2)
        log.info("itens_pedido: %d inflated prices normalised", counts["inflated_price"])

    # anomaly 9: recompute valor_mercadoria consistently
    expected = (df["quantidade_expedida"] * df["preco_unitario_brl"]
                * (1 - df["desconto_pct"])).round(2)
    mismatched = (df["valor_mercadoria_brl"] - expected).abs() > 0.01
    counts["discount_not_applied"] = int(mismatched.sum())
    if counts["discount_not_applied"]:
        df.loc[mismatched, "valor_mercadoria_brl"] = expected[mismatched]
        log.info("itens_pedido: %d values recomputed (discount not applied upstream)",
                 counts["discount_not_applied"])

    return df.drop(columns=["_markup", "categoria", "custo_referencia_brl"]), counts


def load_and_validate_all(clean_anomalies: bool = True) -> dict[str, pd.DataFrame]:
    tables = {name: validate(name) for name in RAW_FILES}
    if clean_anomalies:
        tables["produtos"], _ = clean_prices(tables["produtos"])
        tables["itens_pedido"], _ = clean_items(
            tables["itens_pedido"], tables["produtos"])
    return tables
