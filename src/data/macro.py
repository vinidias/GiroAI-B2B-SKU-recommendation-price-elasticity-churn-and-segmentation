"""Macro/commodity price features via agrobr (CEPEA daily spot prices)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import INTERIM_DIR

log = logging.getLogger(__name__)

# Map Giro AI product categories -> CEPEA commodity indicators
CATEGORY_TO_COMMODITY: dict[str, str] = {
    "Bebidas": "cafe",
    "Mercearia": "milho",
    "Limpeza": "algodao",
    "Perecíveis": "boi_gordo",
    "Congelados": "frango_congelado",
    "Higiene": "soja",
}

CEPEA_PRODUCTS = ["soja", "milho", "cafe", "boi_gordo", "frango_congelado", "algodao"]

CACHE_FILE = INTERIM_DIR / "cepea_daily.parquet"


def _fetch_cepea(product: str, start: str = "2023-01-01") -> pd.DataFrame | None:
    try:
        from agrobr.sync import cepea

        df = cepea.indicador(product, inicio=start)
        if df is None or df.empty:
            return None
        df = df.copy()
        df["produto"] = product
        date_col = "data" if "data" in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        value_col = "valor" if "valor" in df.columns else df.columns[1]
        df["valor"] = pd.to_numeric(df[value_col], errors="coerce")
        return df[[date_col, "produto", "valor"]].rename(columns={date_col: "data"})
    except Exception as exc:
        log.warning("CEPEA fetch failed for %s: %s", product, exc)
        return None


def fetch_cepea_daily(products: list[str] | None = None, use_cache: bool = True) -> pd.DataFrame:
    products = products or CEPEA_PRODUCTS
    if use_cache and CACHE_FILE.exists():
        log.info("Loading CEPEA cache from %s", CACHE_FILE)
        return pd.read_parquet(CACHE_FILE)

    frames = []
    for p in products:
        df = _fetch_cepea(p)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        log.warning("No CEPEA data available; returning empty frame")
        return pd.DataFrame(columns=["data", "produto", "valor"])

    out = pd.concat(frames, ignore_index=True).sort_values(["produto", "data"])
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE_FILE, index=False)
    log.info("Cached CEPEA data: %d rows -> %s", len(out), CACHE_FILE)
    return out


def monthly_macro(cepea_daily: pd.DataFrame) -> pd.DataFrame:
    if cepea_daily.empty:
        return pd.DataFrame(columns=["ano_mes", "produto", "preco_commodity"])
    df = cepea_daily.dropna(subset=["data", "valor"]).copy()
    df["ano_mes"] = df["data"].dt.to_period("M").astype(str)
    return (df.groupby(["ano_mes", "produto"], as_index=False)["valor"]
              .mean()
              .rename(columns={"valor": "preco_commodity"}))


def attach_macro_features(elasticity_df: pd.DataFrame,
                          cepea_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join monthly commodity prices onto the elasticity feature frame.

    Gracefully no-ops when CEPEA is unavailable so the model still runs.
    """
    out = elasticity_df.copy()
    if "data_pedido" in out.columns:
        out["ano_mes"] = pd.to_datetime(out["data_pedido"]).dt.to_period("M").astype(str)
    else:
        out["ano_mes"] = pd.NA
    out["commodity"] = out["categoria"].map(CATEGORY_TO_COMMODITY)

    if cepea_daily is None:
        try:
            cepea_daily = fetch_cepea_daily()
        except Exception as exc:
            log.warning("Macro features disabled (CEPEA unavailable): %s", exc)
            cepea_daily = pd.DataFrame()

    macro = monthly_macro(cepea_daily)
    if macro.empty:
        out["preco_commodity"] = np.nan
        out["log_preco_commodity"] = np.nan
        return out

    out = out.merge(
        macro.rename(columns={"produto": "commodity"}),
        on=["ano_mes", "commodity"], how="left",
    )
    out["preco_commodity"] = pd.to_numeric(out["preco_commodity"], errors="coerce")
    out["log_preco_commodity"] = np.log1p(out["preco_commodity"])
    return out

