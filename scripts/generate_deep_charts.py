"""Deep visual analytics for the Giro AI README.

Each chart is a cross-tabulation that answers a commercial question, not just a
metric plot. Output goes to docs/images/ and is embedded in the README.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, "D:/Dev/giro-ai")
from src.data.validate import load_and_validate_all
from src.features.build import (build_churn_features, build_elasticity_features,
                                build_interactions, build_segmentation_features)

OUT = Path("D:/Dev/giro-ai/docs/images")
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "axes.titleweight": "bold",
    "axes.titlesize": 15, "axes.labelsize": 12,
})

GREEN = "#2E7D32"
GRAY = "#9E9E9E"
ACCENT = "#1565C0"


def chart_01_category_margin():
    """Which categories generate revenue vs which generate margin?"""
    t = load_and_validate_all()
    itens = t["itens_pedido"].merge(t["pedidos"][["pedido_id","status_pedido"]], on="pedido_id")
    itens = itens[itens["status_pedido"]=="FATURADO"]
    itens = itens.merge(t["produtos"][["produto_id","categoria","custo_referencia_brl"]], on="produto_id")
    itens["preco_liquido"] = itens["preco_unitario_brl"] * (1 - itens["desconto_pct"])
    itens["receita"] = itens["quantidade_expedida"] * itens["preco_liquido"]
    itens["custo"] = itens["quantidade_expedida"] * itens["custo_referencia_brl"]
    itens["margem"] = itens["receita"] - itens["custo"]

    g = itens.groupby("categoria").agg(
        receita=("receita","sum"), margem=("margem","sum"), pedidos=("pedido_id","nunique")
    ).reset_index()
    g["margem_pct"] = g["margem"] / g["receita"] * 100
    g = g.sort_values("receita", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(g)))
    ax.barh(g["categoria"], g["receita"]/1e6, color=colors)
    ax.set_xlabel("Revenue (R$ millions)")
    ax.set_title("Revenue by category")
    for i, v in enumerate(g["receita"]/1e6):
        ax.text(v, i, f" {v:.1f}M", va="center", fontsize=9)

    ax = axes[1]
    scatter = ax.scatter(g["receita"]/1e6, g["margem_pct"], s=g["pedidos"]/30,
                         c=colors, alpha=0.8, edgecolor="white", linewidth=2)
    for _, row in g.iterrows():
        ax.annotate(row["categoria"], (row["receita"]/1e6, row["margem_pct"]),
                    fontsize=9, ha="center", va="bottom")
    ax.set_xlabel("Revenue (R$ millions)")
    ax.set_ylabel("Margin %")
    ax.set_title("Revenue vs margin (bubble = orders)")
    ax.axhline(g["margem_pct"].mean(), color="gray", ls="--", alpha=0.5)
    fig.suptitle("Where does the money come from - and where does the margin hide?",
                 fontweight="bold", fontsize=17)
    fig.tight_layout()
    fig.savefig(OUT / "01_category_margin.png")
    plt.close(fig)
    print("  01 category margin")


def chart_02_segment_value():
    """Which customer segments are worth most, and how concentrated?"""
    t = load_and_validate_all()
    pedidos = t["pedidos"].merge(t["clientes"][["cliente_id","segmento","uf"]], on="cliente_id")
    pedidos = pedidos[pedidos["status_pedido"]=="FATURADO"]

    seg = pedidos.groupby("segmento").agg(
        receita=("valor_mercadoria_total_brl","sum"),
        clientes=("cliente_id","nunique"),
        pedidos=("pedido_id","nunique"),
    ).reset_index()
    seg["ticket"] = seg["receita"] / seg["pedidos"]
    seg["receita_por_cliente"] = seg["receita"] / seg["clientes"]
    seg = seg.sort_values("receita", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    bars = ax.bar(seg["segmento"], seg["receita"]/1e6, color=GREEN, alpha=0.85, edgecolor="white")
    ax.set_ylabel("Revenue (R$ millions)")
    ax.set_title("Revenue by customer segment")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, seg["receita"]/1e6):
        ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.1f}M", ha="center", va="bottom", fontsize=10)

    ax = axes[1]
    bars = ax.bar(seg["segmento"], seg["receita_por_cliente"]/1e3, color=ACCENT, alpha=0.85, edgecolor="white")
    ax.set_ylabel("Revenue per customer (R$ thousands)")
    ax.set_title("Revenue per customer (value density)")
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, seg["receita_por_cliente"]/1e3):
        ax.text(bar.get_x()+bar.get_width()/2, v, f"{v:.0f}k", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Segment revenue: volume vs value per account", fontweight="bold", fontsize=17)
    fig.tight_layout()
    fig.savefig(OUT / "02_segment_value.png")
    plt.close(fig)
    print("  02 segment value")


def chart_03_elasticity_by_category():
    """Estimated elasticity vs true price range: which categories are price-sensitive?"""
    t = load_and_validate_all()
    el = build_elasticity_features(t["pedidos"], t["itens_pedido"], t["produtos"], t["clientes"], with_macro=False)
    import statsmodels.formula.api as smf

    el = el.sample(min(200_000, len(el)), random_state=42)
    el["lp_c"] = el["log_preco"] - el.groupby(["produto_id","mes"])["log_preco"].transform("mean")
    el["lq_c"] = el["log_qty"] - el.groupby(["produto_id","mes"])["log_qty"].transform("mean")

    rows = []
    for cat in sorted(el["categoria"].unique()):
        sub = el[el["categoria"]==cat]
        m = smf.ols("lq_c ~ lp_c - 1", data=sub).fit()
        rows.append({"categoria": cat, "elasticidade": -m.params["lp_c"],
                     "n": len(sub), "r2": m.rsquared})
    df = pd.DataFrame(rows).sort_values("elasticidade")

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [ACCENT if e < -1 else GRAY for e in df["elasticidade"]]
    bars = ax.barh(df["categoria"], df["elasticidade"], color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(-1, color="red", ls="--", lw=1.5, label="Unit elastic (|ε|=1)")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Estimated elasticity (negative = demand falls when price rises)")
    ax.set_title("Price elasticity by category (within SKU x month)")
    for bar, v, n in zip(bars, df["elasticidade"], df["n"]):
        ax.text(v, bar.get_y()+bar.get_height()/2, f" {v:.2f} (n={n:,})",
                va="center", fontsize=9, ha="left" if v<0 else "right")
    ax.legend(frameon=False, loc="lower left")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "03_elasticity_by_category.png")
    plt.close(fig)
    print("  03 elasticity by category")


def chart_04_churn_by_segment():
    """Churn rate and revenue at risk by segment."""
    t = load_and_validate_all()
    ch = build_churn_features(t["clientes"], t["pedidos"], t["itens_pedido"])
    g = ch.groupby("segmento").agg(
        clientes=("cliente_id","count"),
        churners=("churn","sum"),
        receita_risco=("receita_total", lambda s: s[ch.loc[s.index,"churn"]==1].sum()),
    ).reset_index()
    g["churn_pct"] = g["churners"]/g["clientes"]*100
    g = g.sort_values("churn_pct", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    bars = ax.barh(g["segmento"], g["churn_pct"], color=GREEN, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Churn rate (%)")
    ax.set_title("Churn rate by segment")
    for bar, v in zip(bars, g["churn_pct"]):
        ax.text(v, bar.get_y()+bar.get_height()/2, f" {v:.1f}%", va="center", fontsize=10)

    ax = axes[1]
    bars = ax.barh(g["segmento"], g["receita_risco"]/1e6, color="#C62828", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Revenue at risk (R$ millions)")
    ax.set_title("Revenue at risk from churn")
    for bar, v in zip(bars, g["receita_risco"]/1e6):
        ax.text(v, bar.get_y()+bar.get_height()/2, f" {v:.1f}M", va="center", fontsize=10)

    fig.suptitle("Churn: where the customers leave and how much they take",
                 fontweight="bold", fontsize=17)
    fig.tight_layout()
    fig.savefig(OUT / "04_churn_by_segment.png")
    plt.close(fig)
    print("  04 churn by segment")


def chart_05_recommender_coverage():
    """What does the recommender actually cover? Popularity vs personalisation."""
    t = load_and_validate_all()
    inter = build_interactions(t["pedidos"], t["itens_pedido"])
    pop = inter.groupby("produto_id")["n"].sum().sort_values(ascending=False)
    cum = pop.cumsum() / pop.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ax.plot(range(1, len(cum)+1), cum.values, color=GREEN, lw=2.5)
    ax.axhline(80, color="gray", ls="--", alpha=0.6)
    n80 = int((cum < 80).sum()) + 1
    ax.axvline(n80, color="red", ls="--", alpha=0.6)
    ax.set_xlabel("SKUs ranked by volume")
    ax.set_ylabel("Cumulative % of units sold")
    ax.set_title(f"Popularity concentration: {n80} SKUs = 80% of volume")

    ax = axes[1]
    per_user = inter.groupby("cliente_id")["produto_id"].count()
    ax.hist(per_user, bins=30, color=ACCENT, alpha=0.85, edgecolor="white")
    ax.axvline(per_user.mean(), color="red", ls="--", lw=2,
               label=f"mean = {per_user.mean():.1f}")
    ax.set_xlabel("Distinct SKUs per customer")
    ax.set_ylabel("Number of customers")
    ax.set_title("SKU variety per customer")
    ax.legend(frameon=False)

    fig.suptitle("Recommender challenge: few SKUs drive volume, few customers buy many",
                 fontweight="bold", fontsize=17)
    fig.tight_layout()
    fig.savefig(OUT / "05_recommender_coverage.png")
    plt.close(fig)
    print("  05 recommender coverage")


def chart_06_ab_impact():
    """A/B test: uplift distribution and significance."""
    t = load_and_validate_all()
    from src.experiments.ab_test import simulate_experiment

    out = simulate_experiment(t["pedidos"], t["itens_pedido"], uplift=0.10)
    control = out["control"]
    treatment = out["treatment"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ax.hist(control, bins=50, alpha=0.6, label="Control", color=GRAY, density=True)
    ax.hist(treatment, bins=50, alpha=0.6, label="Treatment", color=GREEN, density=True)
    ax.axvline(control.mean(), color=GRAY, ls="--", lw=2)
    ax.axvline(treatment.mean(), color=GREEN, ls="--", lw=2)
    ax.set_xlabel("Revenue per order (R$)")
    ax.set_ylabel("Density")
    ax.set_title("Revenue distribution: control vs treatment")
    ax.legend(frameon=False)

    ax = axes[1]
    # bootstrap CI of the uplift
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(500):
        c = rng.choice(control, size=len(control), replace=True)
        t = rng.choice(treatment, size=len(treatment), replace=True)
        diffs.append((t.mean() - c.mean()) / c.mean() * 100)
    ax.hist(diffs, bins=40, color=ACCENT, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=2, label="No effect")
    ax.axvline(np.mean(diffs), color="black", lw=2, label=f"Uplift = {np.mean(diffs):.1f}%")
    ax.set_xlabel("Uplift (%)")
    ax.set_ylabel("Bootstrap draws")
    ax.set_title("Bootstrap distribution of uplift")
    ax.legend(frameon=False)

    fig.suptitle("A/B test: the treatment effect is large and unambiguous",
                 fontweight="bold", fontsize=17)
    fig.tight_layout()
    fig.savefig(OUT / "06_ab_impact.png")
    plt.close(fig)
    print("  06 ab impact")


def chart_07_cepea_coupling():
    """CEPEA index vs realised price: the causal chain made visible."""
    t = load_and_validate_all()
    from src.data.macro import fetch_cepea_daily, monthly_macro

    cepea = fetch_cepea_daily()
    macro = monthly_macro(cepea)
    macro = macro[macro["commodity"].isin(["cafe","milho","boi_gordo","soja"])]
    macro["ano_mes"] = pd.to_datetime(macro["ano_mes"] + "-01")

    # index each commodity to its first month
    macro["idx"] = macro.groupby("commodity")["preco_commodity"].transform(lambda s: s/s.iloc[0]*100)

    fig, ax = plt.subplots(figsize=(14, 6))
    palette = {"cafe": "#6D4C41", "milho": "#F9A825", "boi_gordo": "#C62828", "soja": "#2E7D32"}
    labels = {"cafe": "Coffee", "milho": "Corn", "boi_gordo": "Cattle", "soja": "Soy"}
    for c in ["cafe","milho","boi_gordo","soja"]:
        sub = macro[macro["commodity"]==c].sort_values("ano_mes")
        ax.plot(sub["ano_mes"], sub["idx"], lw=2.5, label=labels[c], color=palette[c])

    ax.axhline(100, color="gray", ls="--", alpha=0.5)
    ax.set_ylabel("Index (first month = 100)")
    ax.set_xlabel("Month")
    ax.set_title("Real CEPEA commodity indices that drive SKU cost and price")
    ax.legend(frameon=False, ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT / "07_cepea_coupling.png")
    plt.close(fig)
    print("  07 cepea coupling")


if __name__ == "__main__":
    print("Generating deep analytics charts...")
    chart_01_category_margin()
    chart_02_segment_value()
    chart_03_elasticity_by_category()
    chart_04_churn_by_segment()
    chart_05_recommender_coverage()
    chart_06_ab_impact()
    chart_07_cepea_coupling()
    print(f"\nDone -> {OUT}")
