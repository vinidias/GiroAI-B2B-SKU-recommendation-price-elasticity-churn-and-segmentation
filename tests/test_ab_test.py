import numpy as np

from src.experiments.ab_test import simulate_experiment, two_proportion_test, welch_ttest
from src.data.synthetic import generate


def test_two_proportion_significance():
    r = two_proportion_test(control_success=100, control_n=1000,
                            treatment_success=150, treatment_n=1000)
    assert r.significant
    assert r.uplift_pct > 0


def test_two_proportion_no_signal():
    r = two_proportion_test(control_success=100, control_n=1000,
                            treatment_success=101, treatment_n=1000)
    assert not r.significant


def test_welch():
    rng = np.random.default_rng(0)
    c = rng.normal(100, 10, 500)
    t = rng.normal(110, 10, 500)
    r = welch_ttest(c, t)
    assert r.significant
    assert r.treatment_mean > r.control_mean


def test_simulate_experiment():
    tables = generate(n_customers=100, n_products=30, n_orders=1000, days=365)
    out = simulate_experiment(tables["pedidos"], tables["itens_pedido"], uplift=0.10)
    assert "result" in out and "frame" in out
    assert out["result"].treatment_mean >= out["result"].control_mean
