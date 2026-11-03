"""Data-quality tests against the real Atacado Atlas tables.

These run on the CSVs in data/raw (the Atlas release). They assert the loader
handles the controlled anomalies correctly and that Pandera schemas pass.
"""
import pandas as pd
import pytest

from src.data.validate import PRIMARY_KEYS, SCHEMAS, load_and_validate_all, load_raw


@pytest.fixture(scope="module")
def tables():
    return load_and_validate_all()


def test_all_tables_loaded(tables):
    for name in SCHEMAS:
        assert name in tables, f"missing {name}"
        assert len(tables[name]) > 0


def test_schemas_validate(tables):
    # load_and_validate_all already validates; re-run to be explicit
    for name, schema in SCHEMAS.items():
        schema.validate(tables[name])


def test_primary_keys_unique(tables):
    for name, key in PRIMARY_KEYS.items():
        assert tables[name][key].is_unique, f"{name}.{key} not unique"


def test_deduplication_matches_manifesto():
    # The Atlas manifesto states the expected clean counts:
    #   pedidos.csv      52422 -> 52397
    #   itens_pedido.csv 147082 -> 147053
    # (duplicates, a bad customer and one empty primary key account for the gap)
    raw_p = pd.read_csv("data/raw/pedidos.csv", encoding="utf-8-sig", dtype=str)
    raw_i = pd.read_csv("data/raw/itens_pedido.csv", encoding="utf-8-sig", dtype=str)
    assert len(raw_p) == 52422
    assert len(raw_i) == 147082
    assert len(load_raw("pedidos")) == 52397
    assert len(load_raw("itens_pedido")) == 147053


def test_status_normalized(tables):
    valid = {"FATURADO", "CANCELADO", "SEM_EXPEDICAO", "BLOQUEADO_CREDITO"}
    assert set(tables["pedidos"]["status_pedido"].unique()).issubset(valid)


def test_discount_is_fraction(tables):
    d = tables["itens_pedido"]["desconto_pct"]
    assert d.between(0, 1).all()


def test_no_nat_dates(tables):
    assert tables["pedidos"]["data_pedido"].notna().all()
