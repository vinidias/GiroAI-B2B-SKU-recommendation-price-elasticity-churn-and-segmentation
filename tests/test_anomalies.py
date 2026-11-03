"""Regression tests for the 14 Atlas anomalies.

Each test asserts that the loader produces the clean invariant, so a future
change to validate.py cannot silently reintroduce a defect.
"""
import pandas as pd
import pytest

from src.data.validate import MARKUP_SANITY_MAX, load_and_validate_all


@pytest.fixture(scope="module")
def tables():
    return load_and_validate_all()


def test_01_row_counts_match_manifesto(tables):
    assert len(tables["pedidos"]) == 52397
    assert len(tables["itens_pedido"]) == 147053


def test_02_no_orphan_customers(tables):
    known = set(tables["clientes"]["cliente_id"])
    assert set(tables["pedidos"]["cliente_id"]).issubset(known)


def test_03_no_orphan_products(tables):
    known = set(tables["produtos"]["produto_id"])
    assert set(tables["itens_pedido"]["produto_id"]).issubset(known)
    assert "P999" not in known


def test_04_no_junk_in_numeric_columns(tables):
    it = tables["itens_pedido"]
    assert it["preco_unitario_brl"].notna().all()
    assert it["quantidade_expedida"].notna().all()


def test_05_no_duplicate_primary_keys(tables):
    for name, key in [("pedidos", "pedido_id"), ("itens_pedido", "item_pedido_id"),
                      ("clientes", "cliente_id"), ("produtos", "produto_id")]:
        assert tables[name][key].is_unique, f"{name}.{key} not unique"


def test_06_categoricals_uppercased(tables):
    for name, col in [("pedidos", "status_pedido"), ("clientes", "segmento"),
                      ("produtos", "categoria")]:
        vals = tables[name][col].dropna().unique()
        assert all(v == v.upper() for v in vals), f"{name}.{col} not uppercase"


def test_07_discount_is_fraction(tables):
    d = tables["itens_pedido"]["desconto_pct"]
    assert d.between(0, 1).all()
    assert d.max() <= 0.08  # the Atlas tops out at 8%


def test_08_no_decimal_comma_left(tables):
    it = tables["itens_pedido"]
    assert it["preco_unitario_brl"].notna().all()
    assert (it["preco_unitario_brl"] > 0).all()


def test_09_price_inflation_normalised(tables):
    it = tables["itens_pedido"].merge(
        tables["produtos"][["produto_id", "custo_referencia_brl"]], on="produto_id")
    markup = it["preco_unitario_brl"] / it["custo_referencia_brl"]
    assert (markup <= MARKUP_SANITY_MAX).all(), "box-conversion inflation came back"


def test_10_discount_applied_to_value(tables):
    it = tables["itens_pedido"]
    expected = (it["quantidade_expedida"] * it["preco_unitario_brl"]
                * (1 - it["desconto_pct"])).round(2)
    diff = (it["valor_mercadoria_brl"] - expected).abs()
    assert (diff <= 0.01).all(), "valor_mercadoria inconsistent with discount"


def test_11_no_negative_quantities(tables):
    assert (tables["itens_pedido"]["quantidade_solicitada"] >= 0).all()
    assert (tables["itens_pedido"]["quantidade_expedida"] >= 0).all()


def test_12_dates_parse_and_in_range(tables):
    p = tables["pedidos"]
    assert p["data_pedido"].notna().all()
    assert p["data_pedido"].min() >= pd.Timestamp("2024-09-01")
    assert p["data_pedido"].max() <= pd.Timestamp("2026-09-01")


def test_13_optional_nulls_are_expected(tables):
    """Blanks that are legitimate, and blanks that are not.

    validade_dias / temperatura_max_c are blank for non-perishables (fine).
    frete_rateado_brl is mostly blank for non-invoiced orders, but 278 invoiced
    orders also carry a null freight - documented here rather than silently
    ignored, since it affects any freight-based feature.
    """
    p = tables["produtos"]
    assert (p["familia"] == "NAO_PERECIVEL").sum() > 0

    ped = tables["pedidos"]
    blank = ped[ped["frete_rateado_brl"].isna()]
    invoiced_blank = blank[blank["status_pedido"] == "FATURADO"]
    assert len(blank) > 0
    # The null freight on invoiced orders is a known Atlas quirk; assert the
    # magnitude stays small so it does not silently grow.
    assert len(invoiced_blank) < len(ped) * 0.01


def test_14_price_never_zero_or_negative(tables):
    assert (tables["itens_pedido"]["preco_unitario_brl"] > 0).all()
    assert (tables["produtos"]["custo_referencia_brl"] >= 0).all()
    assert (tables["produtos"]["preco_lista_brl"] >= 0).all()
