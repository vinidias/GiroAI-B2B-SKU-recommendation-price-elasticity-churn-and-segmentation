"""Log-log price elasticity regression with per-segment elasticities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit_elasticity(df: pd.DataFrame, segment_col: str = "categoria",
                   use_macro: bool = True, fixed_effects: bool = True):
    """Estimate price elasticity with a within-SKU x month fixed-effects OLS.

    Why fixed effects: the pooled specification (log_qty ~ log_preco * C(cat))
    conflates three sources of price variation - the price table assigned to the
    customer, monthly list price changes and per-order promotions. Only the
    last one is the demand response we want to measure. Absorbing SKU x month
    means keeps only within-cell variation, which is exactly the promotion.

    The category elasticities reported are the pooled within-SKU slopes for each
    category's SKUs.
    """
    import numpy as np

    work = df.copy()
    work["log_preco"] = work["log_preco"].astype(float)
    work["log_qty"] = work["log_qty"].astype(float)

    if fixed_effects and "produto_id" in work.columns and "mes" in work.columns:
        g = work.groupby(["produto_id", "mes"])
        work["_lp"] = work["log_preco"] - g["log_preco"].transform("mean")
        work["_lq"] = work["log_qty"] - g["log_qty"].transform("mean")
        formula = "_lq ~ _lp - 1"
        model = smf.ols(formula, data=work).fit(cov_type="HC3")
        pooled = float(model.params.get("_lp", np.nan))
        rows = []
        for cat in sorted(work[segment_col].dropna().unique()):
            sub = work[work[segment_col] == cat]
            m = smf.ols("_lq ~ _lp - 1", data=sub).fit(cov_type="HC3")
            rows.append({segment_col: cat, "elasticidade": float(m.params.get("_lp", np.nan)),
                         "p_valor": float(m.pvalues.get("_lp", 1.0)),
                         "n": int(len(sub))})
        return model, pd.DataFrame(rows).sort_values("elasticidade")

    base = f"log_qty ~ log_preco * C({segment_col})"
    macro_terms = ""
    if use_macro and "log_preco_commodity" in df.columns and df["log_preco_commodity"].notna().any():
        macro_terms += " + log_preco_commodity"
    formula = base + macro_terms + " + margem_unit"
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

