"""Feature-builder tests against the real Atlas tables."""
import pytest

from src.data.validate import load_and_validate_all
from src.features.build import (
    build_churn_features,
    build_elasticity_features,
    build_interactions,
    build_segmentation_features,
)


@pytest.fixture(scope="module")
def tables():
    return load_and_validate_all()


def test_elasticity_features(tables):
    df = build_elasticity_features(tables["pedidos"], tables["itens_pedido"],
                                   tables["produtos"], tables["clientes"],
                                   with_macro=False)
    assert len(df) > 0
    assert df["log_preco"].notna().all()
    assert df["log_qty"].notna().all()
    assert df["categoria"].nunique() == 12


def test_churn_features(tables):
    df = build_churn_features(tables["clientes"], tables["pedidos"], tables["itens_pedido"])
    assert "churn" in df.columns
    assert df["churn"].isin([0, 1]).all()
    assert len(df) == len(tables["clientes"])
    assert 0 < df["churn"].mean() < 0.5


def test_segmentation_features(tables):
    df = build_segmentation_features(tables["clientes"], tables["pedidos"], tables["itens_pedido"])
    assert len(df) == len(tables["clientes"])
    assert (df[["recencia", "frequencia", "monetario"]] >= 0).all().all()


def test_interactions(tables):
    df = build_interactions(tables["pedidos"], tables["itens_pedido"])
    assert {"cliente_id", "produto_id", "n"}.issubset(df.columns)
    assert (df["n"] > 0).all()
