"""Macro/commodity price features via agrobr.

Sources (all optional, graceful fallback):
  - CEPEA  -> daily spot prices for 6 commodities (fetched, cached)
  - USDA PSD -> annual supply/demand balance per commodity (requires API key)
  - CONAB CEASA -> wholesale produce prices at Brazilian CEASAs

Features are joined to the elasticity frame on month + commodity. When any
source is unavailable, the corresponding columns are NaN and the model
skips those terms (see fit_elasticity(use_macro=...)).
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from src.config import INTERIM_DIR

log = logging.getLogger(__name__)

CATEGORY_TO_COMMODITY: dict[str, str] = {
    "Bebidas": "cafe",
    "Mercearia": "milho",
    "Limpeza": "algodao",
    "Perecíveis": "boi_gordo",
    "Congelados": "frango_congelado",
    "Higiene": "soja",
}

# USDA PSD supports only this subset (see agrobr usda catalogue).
# Categories without a USDA counterpart (boi_gordo, frango_congelado) get
# NaN for producao_usda — the elasticity model tolerates that.
USDA_COMMODITY = {
    "cafe": "cafe",
    "milho": "milho",
    "algodao": "algodao",
    "soja": "soja",
}

CEPEA_PRODUCTS = ["soja", "milho", "cafe", "boi_gordo", "frango_congelado", "algodao"]

CEPEA_CACHE = INTERIM_DIR / "cepea_daily.parquet"
USDA_CACHE = INTERIM_DIR / "usda_psd.parquet"
CEASA_CACHE = INTERIM_DIR / "conab_ceasa.parquet"


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
    """Return long DataFrame [data, produto, valor] of daily CEPEA prices."""
    products = products or CEPEA_PRODUCTS
    if use_cache and CEPEA_CACHE.exists():
        log.info("Loading CEPEA cache from %s", CEPEA_CACHE)
        return pd.read_parquet(CEPEA_CACHE)

    frames = [df for p in products if (df := _fetch_cepea(p)) is not None and not df.empty]
    if not frames:
        log.warning("No CEPEA data available; returning empty frame")
        return pd.DataFrame(columns=["data", "produto", "valor"])

    out = pd.concat(frames, ignore_index=True).sort_values(["produto", "data"])
    CEPEA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CEPEA_CACHE, index=False)
    log.info("Cached CEPEA data: %d rows -> %s", len(out), CEPEA_CACHE)
    return out


def monthly_macro(cepea_daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily CEPEA prices to monthly mean: [ano_mes, commodity, preco_commodity]."""
    if cepea_daily.empty:
        return pd.DataFrame(columns=["ano_mes", "commodity", "preco_commodity"])
    df = cepea_daily.dropna(subset=["data", "valor"]).copy()
    df["ano_mes"] = df["data"].dt.to_period("M").astype(str)
    return (df.groupby(["ano_mes", "produto"], as_index=False)["valor"]
              .mean()
              .rename(columns={"produto": "commodity", "valor": "preco_commodity"}))


def fetch_usda_psd(commodities: list[str] | None = None,
                   market_years: list[int] | None = None,
                   use_cache: bool = True) -> pd.DataFrame:
    """Return long DataFrame [commodity, market_year, attribute, value].

    Defaults to the current year and the four preceding ones so the feature
    frame can be joined onto orders from earlier years.
    """
    if use_cache and USDA_CACHE.exists():
        log.info("Loading USDA cache from %s", USDA_CACHE)
        return pd.read_parquet(USDA_CACHE)

    if not os.getenv("AGROBR_USDA_API_KEY"):
        log.warning("AGROBR_USDA_API_KEY not set; skipping USDA PSD")
        return pd.DataFrame(columns=["commodity", "market_year", "attribute", "value"])

    if market_years is None:
        current_year = pd.Timestamp.today().year
        market_years = list(range(current_year - 4, current_year + 1))

    slugs = commodities or list(USDA_COMMODITY.values())
    frames = []
    for slug in slugs:
        for year in market_years:
            try:
                from agrobr.sync import usda

                df = usda.psd(slug, country="BR", market_year=year)
                if df is None or df.empty:
                    continue
                df = df.copy()
                df["commodity"] = slug
                frames.append(df[["commodity", "market_year", "attribute", "value"]])
            except Exception as exc:
                log.warning("USDA PSD fetch failed for %s/%s: %s", slug, year, exc)

    if not frames:
        return pd.DataFrame(columns=["commodity", "market_year", "attribute", "value"])
    out = pd.concat(frames, ignore_index=True)
    USDA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(USDA_CACHE, index=False)
    log.info("Cached USDA data: %d rows -> %s", len(out), USDA_CACHE)
    return out


def usda_production_index(usda_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot USDA annual PSD into [commodity, market_year, producao_usda]."""
    if usda_df.empty:
        return pd.DataFrame(columns=["commodity", "market_year", "producao_usda"])
    prod = usda_df[usda_df["attribute"].str.lower().eq("production")].copy()
    if prod.empty:
        return pd.DataFrame(columns=["commodity", "market_year", "producao_usda"])
    return (prod.groupby(["commodity", "market_year"], as_index=False)["value"]
                .mean()
                .rename(columns={"value": "producao_usda"}))


def fetch_ceasa(produtos: list[str] | None = None,
                ceasa: str = "SAO PAULO",
                use_cache: bool = True) -> pd.DataFrame:
    """Return long DataFrame [data, produto, preco] of CEASA wholesale prices."""
    if use_cache and CEASA_CACHE.exists():
        log.info("Loading CEASA cache from %s", CEASA_CACHE)
        return pd.read_parquet(CEASA_CACHE)

    produtos = produtos or ["tomate", "batata", "cebola", "laranja", "banana"]
    frames = []
    for p in produtos:
        try:
            from agrobr.sync import conab

            df = conab.ceasa_precos(produto=p, ceasa=ceasa)
            if df is None or df.empty:
                continue
            df = df.copy()
            df["data"] = pd.to_datetime(df["data"], errors="coerce")
            frames.append(df[["data", "produto", "preco"]])
        except Exception as exc:
            log.warning("CEASA fetch failed for %s: %s", p, exc)

    if not frames:
        return pd.DataFrame(columns=["data", "produto", "preco"])
    out = pd.concat(frames, ignore_index=True)
    CEASA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CEASA_CACHE, index=False)
    log.info("Cached CEASA data: %d rows -> %s", len(out), CEASA_CACHE)
    return out


def attach_macro_features(elasticity_df: pd.DataFrame,
                          cepea_daily: pd.DataFrame | None = None,
                          usda_df: pd.DataFrame | None = None,
                          ceasa_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join monthly CEPEA prices + annual USDA production onto the elasticity frame.

    Adds columns: ano_mes, commodity, preco_commodity, log_preco_commodity,
    producao_usda, log_producao_usda.
    """
    out = elasticity_df.copy()
    if "data_pedido" in out.columns:
        out["ano_mes"] = pd.to_datetime(out["data_pedido"]).dt.to_period("M").astype(str)
        out["market_year"] = pd.to_datetime(out["data_pedido"]).dt.year
    else:
        out["ano_mes"] = pd.NA
        out["market_year"] = pd.NA
    out["commodity"] = out["categoria"].map(CATEGORY_TO_COMMODITY)

    if cepea_daily is None:
        try:
            cepea_daily = fetch_cepea_daily()
        except Exception as exc:
            log.warning("CEPEA unavailable: %s", exc)
            cepea_daily = pd.DataFrame()

    macro = monthly_macro(cepea_daily)
    if macro.empty:
        out["preco_commodity"] = np.nan
        out["log_preco_commodity"] = np.nan
    else:
        out = out.merge(macro, on=["ano_mes", "commodity"], how="left")
        out["preco_commodity"] = pd.to_numeric(out["preco_commodity"], errors="coerce")
        out["log_preco_commodity"] = np.log1p(out["preco_commodity"])

    if usda_df is None:
        try:
            usda_df = fetch_usda_psd()
        except Exception as exc:
            log.warning("USDA unavailable: %s", exc)
            usda_df = pd.DataFrame()

    prod = usda_production_index(usda_df)
    if prod.empty:
        out["producao_usda"] = np.nan
        out["log_producao_usda"] = np.nan
    else:
        prod = prod.rename(columns={"commodity": "usda_slug"})
        out["usda_slug"] = out["commodity"].map(USDA_COMMODITY)
        # Strict exact-year merge only. A per-category fallback would make
        # producao_usda constant within each category/year, which is collinear
        # with the category dummies and produces a rank-deficient design matrix.
        out = out.merge(prod, on=["usda_slug", "market_year"], how="left")
        out["producao_usda"] = pd.to_numeric(out["producao_usda"], errors="coerce")
        out["log_producao_usda"] = np.log1p(out["producao_usda"])

    return out
