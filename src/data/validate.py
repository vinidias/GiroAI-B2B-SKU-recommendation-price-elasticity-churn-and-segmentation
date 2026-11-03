"""Pandera schemas + loader for the real Atacado Atlas tables.

Column names and file layout mirror the Atlas release (22 CSVs, 730 days,
144 SKUs, 480 customers).

The Atlas intentionally ships controlled data-quality anomalies (README and
manifesto.csv document them):

  - exact duplicate rows (pedidos 52422 -> 52397, itens_pedido 147082 -> 147053)
  - a duplicate order pointing at a non-existent customer (C9999)
  - textual junk in numeric columns (e.g. "INVALIDO" in preco_unitario_brl)
  - 4 duplicated item_pedido_id rows
  - box-conversion price inflation: CX SKUs listed at 100-450x cost, stored both
    in produtos.preco_lista_brl and itens_pedido.preco_unitario_brl (25,555 of
    147,042 items, 17%). Left in, DESCARTAVEIS alone reports R$162.8M of phantom
    revenue. clean_prices() and clean_item_prices() normalise them to the
    category's median markup.

The loader therefore:
  1. reads every column as string (avoids pandas silently turning junk into NaN),
  2. drops exact duplicates and duplicate business keys (keep first),
  3. coerces critical numeric/date columns and drops rows that fail,
  4. validates the cleaned frame against a Pandera schema.

Every removed row is counted and logged so the cleaning is auditable.
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

# Categorical columns normalized to UPPER + stripped. The Atlas injects
# capitalization anomalies (Faturado vs FATURADO, Mercearia vs MERCEARIA).
CATEGORICAL_COLS = {
    "pedidos": ["status_pedido", "canal_pedido", "tabela_preco_aplicada", "moeda"],
    "clientes": ["segmento", "uf", "situacao"],
    "produtos": ["categoria", "familia", "unidade_venda", "unidade_estoque", "situacao"],
    "vendedores": ["regiao", "situacao"],
    "tabelas_preco": ["segmento"],
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

# Rows that fail coercion on these columns are dropped (they are unusable).
CRITICAL_COLS = {
    "pedidos": ["data_pedido", "valor_mercadoria_total_brl"],
    "itens_pedido": ["pedido_id", "produto_id", "quantidade_solicitada",
                     "preco_unitario_brl"],
    "produtos": ["custo_referencia_brl", "preco_lista_brl"],
    "precos_produto": ["preco_brl", "vigencia_inicio"],
    "tabelas_preco": ["fator_sobre_lista"],
}

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
        "quantidade_solicitada": Column(int, Check.ge(1)),
        "quantidade_expedida": Column(int, Check.ge(0)),
        "preco_unitario_brl": Column(float, Check.gt(0)),
        "desconto_pct": Column(float, Check.in_range(0, 1)),  # fraction after normalization
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


MARKUP_SANITY_MAX = 50.0  # Atlas inflates some CX prices by 100-450x


def _coerce(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS.get(name, []):
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper()
    for col in DATE_COLS.get(name, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in NUMERIC_COLS.get(name, []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Atlas stores desconto_pct as a percentage (0-8); normalize to fraction.
    if name == "itens_pedido" and "desconto_pct" in df.columns:
        df["desconto_pct"] = pd.to_numeric(df["desconto_pct"], errors="coerce") / 100.0
    return df


def load_raw(name: str, data_dir: Path | None = None,
             drop_duplicates: bool = True) -> pd.DataFrame:
    """Load one Atlas table, cleaning controlled anomalies.

    Reads all columns as string first so pandas never silently turns junk
    values into NaN before deduplication. Then drops duplicates (exact and by
    business key), coerces types, and drops rows that fail on critical columns.
    """
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


def clean_item_prices(itens: pd.DataFrame, produtos: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Normalise inflated per-item prices using the category's median markup.

    The Atlas stores the box-conversion artefact directly in preco_unitario_brl
    for 25,555 of 147,042 items (17%). Left in, DESCARTAVEIS alone reports
    R$162.8M of revenue that never existed. We recompute the item price as
    custo_referencia * category_median_markup when the original markup exceeds
    MARKUP_SANITY_MAX, preserving discounts.

    Returns the cleaned frame and how many rows were fixed.
    """
    df = itens.merge(
        produtos[["produto_id", "categoria", "custo_referencia_brl"]],
        on="produto_id", how="left",
    )
    df["_markup"] = df["preco_unitario_brl"] / df["custo_referencia_brl"].replace(0, np.nan)
    sane = df[df["_markup"] <= MARKUP_SANITY_MAX]
    cat_markup = sane.groupby("categoria")["_markup"].median()
    bad = df["_markup"] > MARKUP_SANITY_MAX
    n_fixed = int(bad.sum())
    if n_fixed:
        new_price = (df.loc[bad, "custo_referencia_brl"]
                     * df.loc[bad, "categoria"].map(cat_markup)).round(2)
        df.loc[bad, "preco_unitario_brl"] = new_price
        df.loc[bad, "valor_mercadoria_brl"] = (
            df.loc[bad, "quantidade_expedida"] * new_price * (1 - df.loc[bad, "desconto_pct"])
        ).round(2)
        log.info("itens_pedido: %d items with inflated prices normalised", n_fixed)
    return df.drop(columns=["_markup", "categoria", "custo_referencia_brl"]), n_fixed


def clean_prices(produtos: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Flag Atlas SKUs whose list price is a box-conversion artefact.

    The Atlas multiplies the unit price by the box factor twice for some CX
    SKUs, producing markups of 100-450x (a R$7.24 box of disposable plates is
    listed at R$3,292.50). Left in, these SKUs dominate every revenue chart.

    Strategy: compute markup = preco_lista / custo. When markup exceeds
    MARKUP_SANITY_MAX, recompute preco_lista as custo * median markup of the
    same category's sane SKUs. Returns the cleaned frame and how many were fixed.
    """
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


def load_and_validate_all(clean_price_anomalies: bool = True) -> dict[str, pd.DataFrame]:
    tables = {name: validate(name) for name in RAW_FILES}
    if clean_price_anomalies:
        tables["produtos"], _ = clean_prices(tables["produtos"])
        tables["itens_pedido"], _ = clean_item_prices(
            tables["itens_pedido"], tables["produtos"])
    return tables
