"""Macro-feature tests (hermetic — no network, no Atlas dependency)."""
import pandas as pd
import pytest

from src.data.macro import CATEGORY_TO_COMMODITY, monthly_macro, usda_production_index
from src.data.validate import load_and_validate_all
from src.features.build import build_elasticity_features


@pytest.fixture(scope="module")
def macro_frame():
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    rows = []
    for prod in set(CATEGORY_TO_COMMODITY.values()):
        rows.append(pd.DataFrame({
            "data": dates,
            "produto": prod,
            "valor": 100.0 + (dates.day_of_year.values * 0.1),
        }))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def usda_frame():
    return pd.DataFrame({
        "commodity": ["soja", "milho", "cafe", "algodao"],
        "market_year": [2024] * 4,
        "attribute": ["Production"] * 4,
        "value": [150_000.0, 120_000.0, 3_500.0, 2_800.0],
    })


def test_category_mapping_covers_all_atlas_categories():
    atlas = {"MERCEARIA", "BEBIDAS", "FRIOS", "PADARIA", "LATICINIOS",
             "HIGIENE", "LIMPEZA", "CONGELADOS", "SNACKS", "HORTIFRUTI",
             "PET", "DESCARTAVEIS"}
    assert atlas.issubset(set(CATEGORY_TO_COMMODITY))


def test_monthly_macro_aggregates(macro_frame):
    m = monthly_macro(macro_frame)
    assert set(m.columns) == {"ano_mes", "commodity", "preco_commodity"}
    assert m["ano_mes"].nunique() == 12


def test_usda_production_index(usda_frame):
    prod = usda_production_index(usda_frame)
    assert set(prod.columns) == {"commodity", "market_year", "producao_usda"}
    assert len(prod) == 4


def test_attach_macro_adds_columns(macro_frame, usda_frame):
    tables = load_and_validate_all()
    df = build_elasticity_features(tables["pedidos"], tables["itens_pedido"],
                                   tables["produtos"], tables["clientes"],
                                   cepea_daily=macro_frame, usda_df=usda_frame)
    assert "log_preco_commodity" in df.columns
    assert "log_producao_usda" in df.columns


def test_attach_macro_graceful_when_missing():
    tables = load_and_validate_all()
    df = build_elasticity_features(tables["pedidos"], tables["itens_pedido"],
                                   tables["produtos"], tables["clientes"],
                                   cepea_daily=pd.DataFrame(columns=["data", "produto", "valor"]),
                                   usda_df=pd.DataFrame(columns=["commodity", "market_year", "attribute", "value"]))
    assert df["log_preco_commodity"].isna().all()
    assert df["log_producao_usda"].isna().all()
