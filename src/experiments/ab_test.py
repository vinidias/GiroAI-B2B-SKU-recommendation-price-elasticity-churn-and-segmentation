"""A/B test simulation with control group and significance testing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.config import RANDOM_STATE


@dataclass
class ABResult:
    control_mean: float
    treatment_mean: float
    uplift_pct: float
    p_value: float
    ci_low: float
    ci_high: float
    significant: bool


def two_proportion_test(control_success: int, control_n: int, treatment_success: int, treatment_n: int,
                        alpha: float = 0.05) -> ABResult:
    p_c = control_success / control_n
    p_t = treatment_success / treatment_n
    pooled = (control_success + treatment_success) / (control_n + treatment_n)
    se = np.sqrt(pooled * (1 - pooled) * (1 / control_n + 1 / treatment_n))
    z = (p_t - p_c) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    diff = p_t - p_c
    se_diff = np.sqrt(p_c * (1 - p_c) / control_n + p_t * (1 - p_t) / treatment_n)
    return ABResult(control_mean=p_c, treatment_mean=p_t,
                    uplift_pct=(diff / p_c * 100) if p_c else 0.0,
                    p_value=float(p_value), ci_low=float(diff - 1.96 * se_diff),
                    ci_high=float(diff + 1.96 * se_diff), significant=bool(p_value < alpha))


def welch_ttest(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05) -> ABResult:
    _, p = stats.ttest_ind(treatment, control, equal_var=False)
    diff = treatment.mean() - control.mean()
    se = np.sqrt(treatment.var(ddof=1) / len(treatment) + control.var(ddof=1) / len(control))
    return ABResult(control_mean=float(control.mean()), treatment_mean=float(treatment.mean()),
                    uplift_pct=float(diff / control.mean() * 100) if control.mean() else 0.0,
                    p_value=float(p), ci_low=float(diff - 1.96 * se), ci_high=float(diff + 1.96 * se),
                    significant=bool(p < alpha))


def simulate_experiment(pedidos, itens, treatment_share: float = 0.5, uplift: float = 0.05,
                        seed: int = RANDOM_STATE) -> dict:
    rng = np.random.default_rng(seed)
    cust = pedidos["cliente_id"].unique()
    mask = rng.random(len(cust)) < treatment_share
    treat_ids = set(cust[mask])
    pedidos = pedidos.copy()
    pedidos["group"] = pedidos["cliente_id"].apply(
        lambda c: "treatment" if c in treat_ids else "control")
    totals = (itens.groupby("pedido_id")["valor_mercadoria_brl"]
                   .sum().rename("receita").reset_index())
    df = pedidos.merge(totals, on="pedido_id", how="left").fillna({"receita": 0.0})
    df = df[df["status_pedido"] == "FATURADO"]
    df["receita_sim"] = np.where(df["group"] == "treatment", df["receita"] * (1 + uplift), df["receita"])
    control = df[df["group"] == "control"]["receita_sim"].to_numpy()
    treat = df[df["group"] == "treatment"]["receita_sim"].to_numpy()
    return {"result": welch_ttest(control, treat), "frame": df, "control": control, "treatment": treat}

