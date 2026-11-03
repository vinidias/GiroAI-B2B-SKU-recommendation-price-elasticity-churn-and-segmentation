import pandas as pd
import pytest

from src.data.macro import CATEGORY_TO_COMMODITY, monthly_macro, usda_production_index
from src.data.synthetic import generate
from src.features.build import build_elasticity_features


@pytest.fixture(scope="module")
def macro_frame():
    """Synthetic CEPEA-like frame — keeps tests hermetic (no network)."""
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    rows = []
    for prod in CATEGORY_TO_COMMODITY.values():
        rows.append(pd.DataFrame({
            "data": dates,
            "produto": prod,
            "valor": 100.0 + (dates.day_of_year.values * 0.1),
        }))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def usda_frame():
    """Synthetic USDA-like frame (production attribute per commodity/year)."""
    return pd.DataFrame({
        "commodity": ["soja", "milho", "cafe", "algodao"],
        "market_year": [2024] * 4,
        "attribute": ["Production"] * 4,
        "value": [150_000.0, 120_000.0, 3_500.0, 2_800.0],
    })


def test_monthly_macro_aggregates(macro_frame):
    m = monthly_macro(macro_frame)
    assert set(m.columns) == {"ano_mes", "commodity", "preco_commodity"}
    assert len(m) == 6 * 12  # 6 commodities x 12 months


def test_usda_production_index(usda_frame):
    prod = usda_production_index(usda_frame)
    assert set(prod.columns) == {"commodity", "market_year", "producao_usda"}
    assert len(prod) == 4
    assert prod["producao_usda"].notna().all()


def test_attach_macro_adds_columns(macro_frame, usda_frame):
    tables = generate(n_customers=20, n_products=10, n_orders=100, days=120)
    df = build_elasticity_features(tables["pedidos"], tables["itens_pedido"],
                                   tables["produtos"], tables["clientes"],
                                   cepea_daily=macro_frame, usda_df=usda_frame)
    assert "preco_commodity" in df.columns
    assert "log_preco_commodity" in df.columns
    assert df["log_preco_commodity"].notna().all()
    # USDA only covers 4 commodities -> some rows should have NaN
    assert "log_producao_usda" in df.columns


def test_attach_macro_graceful_when_missing():
    tables = generate(n_customers=10, n_products=20, n_orders=50, days=60)
    df = build_elasticity_features(tables["pedidos"], tables["itens_pedido"],
                                   tables["produtos"], tables["clientes"],
                                   cepea_daily=pd.DataFrame(columns=["data", "produto", "valor"]),
                                   usda_df=pd.DataFrame(columns=["commodity", "market_year", "attribute", "value"]))
    assert "log_preco_commodity" in df.columns
    assert df["log_preco_commodity"].isna().all()
    assert df["log_producao_usda"].isna().all()
