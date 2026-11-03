"""Pandera schemas + loader for raw tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from src.config import DATA_DIR, RAW_FILES

SCHEMAS: dict[str, DataFrameSchema] = {
    "clientes": DataFrameSchema({
        "cliente_id": Column(int, Check.ge(1), unique=True),
        "nome": Column(str),
        "regiao": Column(str),
        "canal": Column(str),
        "churn": Column(int, Check.isin([0, 1]), required=False),
    }, coerce=True, strict=False),
    "produtos": DataFrameSchema({
        "produto_id": Column(int, Check.ge(1), unique=True),
        "nome": Column(str),
        "categoria": Column(str),
        "custo_unitario": Column(float, Check.ge(0)),
    }, coerce=True, strict=False),
    "pedidos": DataFrameSchema({
        "pedido_id": Column(int, Check.ge(1), unique=True),
        "cliente_id": Column(int, Check.ge(1)),
        "data_pedido": Column("datetime64[ns]"),
        "status": Column(str, Check.isin(["entregue", "cancelado", "devolvido"])),
    }, coerce=True, strict=False),
    "itens_pedido": DataFrameSchema({
        "pedido_id": Column(int, Check.ge(1)),
        "produto_id": Column(int, Check.ge(1)),
        "quantidade": Column(int, Check.ge(1)),
        "preco_unitario": Column(float, Check.gt(0)),
    }, coerce=True, strict=False),
    "precos_produto": DataFrameSchema({
        "produto_id": Column(int, Check.ge(1)),
        "data_vigencia": Column("datetime64[ns]"),
        "preco": Column(float, Check.gt(0)),
    }, coerce=True, strict=False),
    "vendedores": DataFrameSchema({
        "vendedor_id": Column(int, Check.ge(1), unique=True),
        "nome": Column(str),
    }, coerce=True, strict=False),
}


def load_raw(name: str, data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = data_dir or DATA_DIR
    path = data_dir / RAW_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}. Run python -m src.data.synthetic first.")
    df = pd.read_csv(path)
    for col in ("data_pedido", "data_vigencia", "data_cadastro"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def validate(name: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df if df is not None else load_raw(name)
    schema = SCHEMAS.get(name)
    return schema.validate(df) if schema is not None else df


def load_and_validate_all() -> dict[str, pd.DataFrame]:
    return {name: validate(name) for name in RAW_FILES}

