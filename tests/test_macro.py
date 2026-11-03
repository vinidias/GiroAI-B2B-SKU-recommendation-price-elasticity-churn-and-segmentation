import pandas as pd
import pytest

from src.data.macro import (CATEGORY_TO_COMMODITY, attach_macro_features,
                            monthly_macro)
from src.features.build import build_elasticity_features
from src.data.synthetic import generate


@pytest.fixture(scope="module")
def macro_frame():
    """Synthetic CEPEA-like frame — keeps tests hermetic (no network)."""
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    rows = []
    for prod in CATEGORY_TO_COMMODITY.values():
        rows.append(pd.DataFrame({
            "data": dates,
            "produto": prod,
            "valor": 100.0 + (dates.dayofyear.values * 0.1),
        }))
    return pd.concat(rows, ignore_index=True)


def test_monthly_macro_aggregates(macro_frame):
    m = monthly_macro(macro_frame)
    assert set(m.columns) == {"ano_mes", "produto", "preco_commodity"}
    assert len(m) == 6 * 12  # 6 commodities x 12 months


def test_attach_macro_adds_columns(macro_frame):
    tables = generate(n_customers=20, n_products=10, n_orders=100, days=120)
    df = build_elasticity_features(tables["pedidos"], tables["itens_pedido"],
                                   tables["produtos"], tables["clientes"],
                                   cepea_daily=macro_frame)
    assert "preco_commodity" in df.columns
    assert "log_preco_commodity" in df.columns
    assert df["log_preco_commodity"].notna().all()


def test_attach_macro_graceful_when_missing():
    tables = generate(n_customers=10, n_products=5, n_orders=50, days=60)
    df = build_elasticity_features(tables["pedidos"], tables["itens_pedido"],
                                   tables["produtos"], tables["clientes"],
                                   cepea_daily=pd.DataFrame(columns=["data", "produto", "valor"]))
    assert "log_preco_commodity" in df.columns
    # empty CEPEA -> NaN, model would skip macro term
    assert df["log_preco_commodity"].isna().all()

