import pandas as pd
import pytest

from src.data.synthetic import generate
from src.data.validate import SCHEMAS


@pytest.fixture(scope="module")
def tables():
    return generate(n_customers=50, n_products=20, n_orders=400, days=200)


def test_all_tables_generated(tables):
    for name in SCHEMAS:
        assert name in tables, f"missing table {name}"


def test_schemas_validate(tables):
    for name, schema in SCHEMAS.items():
        schema.validate(tables[name])


def test_itens_have_positive_qty(tables):
    assert (tables["itens_pedido"]["quantidade"] > 0).all()


def test_churn_label_binary(tables):
    assert set(tables["clientes"]["churn"].unique()).issubset({0, 1})
