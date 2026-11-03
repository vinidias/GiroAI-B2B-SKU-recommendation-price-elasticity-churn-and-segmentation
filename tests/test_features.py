import pandas as pd
import pytest

from src.data.synthetic import generate
from src.features.build import (build_churn_features, build_elasticity_features,
                                build_interactions, build_segmentation_features)


@pytest.fixture(scope="module")
def tables():
    return generate(n_customers=50, n_products=20, n_orders=400, days=200)


def test_elasticity_features(tables):
    df = build_elasticity_features(tables["pedidos"], tables["itens_pedido"],
                                   tables["produtos"], tables["clientes"])
    assert len(df) > 0
    assert df["log_preco"].notna().all()
    assert df["log_qty"].notna().all()


def test_churn_features(tables):
    df = build_churn_features(tables["clientes"], tables["pedidos"], tables["itens_pedido"])
    assert "churn" in df.columns
    assert df["churn"].isin([0, 1]).all()
    assert len(df) == len(tables["clientes"])


def test_segmentation_features(tables):
    df = build_segmentation_features(tables["clientes"], tables["pedidos"], tables["itens_pedido"])
    assert len(df) == len(tables["clientes"])
    assert (df[["recencia", "frequencia", "monetario"]] >= 0).all().all()


def test_interactions(tables):
    df = build_interactions(tables["pedidos"], tables["itens_pedido"])
    assert {"cliente_id", "produto_id", "n"}.issubset(df.columns)
    assert (df["n"] > 0).all()
