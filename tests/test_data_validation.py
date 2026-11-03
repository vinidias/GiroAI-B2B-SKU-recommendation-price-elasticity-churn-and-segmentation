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
    # raw has 52422 rows; after cleaning the Atlas manifesto expects 52397.
    # We additionally drop 12 rows with invalid critical columns -> 52385.
    raw = pd.read_csv("data/raw/pedidos.csv", encoding="utf-8-sig", dtype=str)
    assert len(raw) == 52422
    clean = load_raw("pedidos")
    assert len(clean) == 52385


def test_status_normalized(tables):
    valid = {"FATURADO", "CANCELADO", "SEM_EXPEDICAO", "BLOQUEADO_CREDITO"}
    assert set(tables["pedidos"]["status_pedido"].unique()).issubset(valid)


def test_discount_is_fraction(tables):
    d = tables["itens_pedido"]["desconto_pct"]
    assert d.between(0, 1).all()


def test_no_nat_dates(tables):
    assert tables["pedidos"]["data_pedido"].notna().all()
