"""Log-log price elasticity regression with per-segment elasticities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit_elasticity(df: pd.DataFrame, segment_col: str = "categoria",
                   use_macro: bool = True):
    base = f"log_qty ~ log_preco * C({segment_col})"
    macro_terms = ""
    if use_macro:
        if "log_preco_commodity" in df.columns and df["log_preco_commodity"].notna().any():
            macro_terms += " + log_preco_commodity"
        if "log_producao_usda" in df.columns and df["log_producao_usda"].notna().any():
            macro_terms += " + log_producao_usda"
    formula = base + macro_terms + " + log_custo + margem_unit"
    model = smf.ols(formula, data=df).fit(cov_type="HC3")
    rows = []
    base = model.params.get("log_preco", np.nan)
    pbase = model.pvalues.get("log_preco", np.nan)
    for i, seg in enumerate(sorted(df[segment_col].dropna().unique().tolist())):
        if i == 0:
            beta, pval = base, pbase
        else:
            key = f"log_preco:C({segment_col})[T.{seg}]"
            beta = base + model.params.get(key, 0.0)
            pval = model.pvalues.get(key, 1.0)
        rows.append({segment_col: seg, "elasticidade": float(beta), "p_valor": float(pval)})
    return model, pd.DataFrame(rows).sort_values("elasticidade")


def predict_demand(model, df: pd.DataFrame) -> np.ndarray:
    return np.expm1(model.predict(df))

