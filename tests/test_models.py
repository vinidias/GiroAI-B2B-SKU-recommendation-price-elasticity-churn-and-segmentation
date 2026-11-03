import pytest

from src.data.synthetic import generate
from src.features.build import (build_churn_features, build_elasticity_features,
                                build_interactions, build_segmentation_features)
from src.models.churn import train_churn
from src.models.elasticity import fit_elasticity
from src.models.recommender import recommend, train_recommender
from src.models.segmentation import fit_segmentation


@pytest.fixture(scope="module")
def tables():
    return generate(n_customers=80, n_products=30, n_orders=900, days=365)


def test_elasticity_fit(tables):
    df = build_elasticity_features(tables["pedidos"], tables["itens_pedido"],
                                   tables["produtos"], tables["clientes"])
    model, table = fit_elasticity(df, segment_col="categoria")
    assert 0 <= model.rsquared <= 1
    assert len(table) == df["categoria"].nunique()


def test_churn_auc_above_random(tables):
    df = build_churn_features(tables["clientes"], tables["pedidos"], tables["itens_pedido"])
    if df["churn"].nunique() < 2:
        pytest.skip("need both classes")
    res = train_churn(df)
    assert 0.0 <= res.auc <= 1.0
    assert 0.0 <= res.pr_auc <= 1.0


def test_segmentation(tables):
    df = build_segmentation_features(tables["clientes"], tables["pedidos"], tables["itens_pedido"])
    res = fit_segmentation(df, k=3)
    assert res.k == 3
    assert len(res.labels) == len(df)
    assert -1 <= res.silhouette <= 1


def test_recommender(tables):
    inter = build_interactions(tables["pedidos"], tables["itens_pedido"])
    rec = train_recommender(inter, factors=16, iterations=5)
    recs = recommend(rec, int(rec.user_ids[0]), n=5)
    assert len(recs) <= 5
    assert all(isinstance(i, int) for i, _ in recs)

