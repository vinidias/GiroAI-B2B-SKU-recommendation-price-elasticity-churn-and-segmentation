"""Clean, publication-style charts for the Giro AI README.

Design principles (borrowed from the Paradime / Isin Pesch school):
  - one message per chart, stated in the title
  - minimal chrome: no top/right spines, no heavy grid, no redundant labels
  - consistent palette across all figures
  - the insight is annotated on the chart, not left to the caption
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, "D:/Dev/giro-ai")
from src.data.validate import load_and_validate_all
from src.experiments.ab_test import simulate_experiment
from src.features.build import (build_churn_features, build_elasticity_features,
                                build_interactions)

OUT = Path("D:/Dev/giro-ai/docs/images")
OUT.mkdir(parents=True, exist_ok=True)

# ---- palette -------------------------------------------------------------
INK = "#1a1a1a"
MUTED = "#6b7280"
GRID = "#e5e7eb"
GREEN = "#16a34a"
RED = "#dc2626"
BLUE = "#2563eb"
AMBER = "#d97706"
CATS = ["#16a34a", "#2563eb", "#d97706", "#dc2626", "#7c3aed",
        "#0891b2", "#65a30d", "#db2777", "#475569", "#0d9488",
        "#b45309", "#4338ca"]

# ---- global style --------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlepad": 14,
    "axes.labelsize": 11,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 10,
})


def _finish(fig, path, subtitle=None):
    """Add a one-line takeaway under the title and save."""
    if subtitle:
        fig.text(0.5, 0.965, subtitle, ha="center", va="top",
                 fontsize=11, color=MUTED, style="italic")
    fig.savefig(OUT / path)
    plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------------------
def chart_01_revenue_margin():
    t = load_and_validate_all()
    it = t["itens_pedido"].merge(t["pedidos"][["pedido_id", "status_pedido"]], on="pedido_id")
    it = it[it["status_pedido"] == "FATURADO"]
    it = it.merge(t["produtos"][["produto_id", "categoria", "custo_referencia_brl"]],
                  on="produto_id")
    it["receita"] = it["quantidade_expedida"] * it["preco_unitario_brl"] * (1 - it["desconto_pct"])
    it["margem"] = it["receita"] - it["quantidade_expedida"] * it["custo_referencia_brl"]

    g = it.groupby("categoria").agg(receita=("receita", "sum"),
                                    margem=("margem", "sum")).reset_index()
    g["receita_M"] = g["receita"] / 1e6
    g["margem_pct"] = g["margem"] / g["receita"] * 100
    g = g.sort_values("receita_M")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    y = np.arange(len(g))
    bars = ax.barh(y, g["receita_M"], color=GREEN, alpha=0.85, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(g["categoria"], fontsize=10)
    ax.set_xlabel("Revenue (R$ millions)")
    ax.set_title("Where the money comes from")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    for bar, (_, row) in zip(bars, g.iterrows()):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"R$ {row.receita_M:.1f}M  ·  {row.margem_pct:.0f}% margin",
                va="center", fontsize=9, color=INK)

    ax.set_xlim(0, g["receita_M"].max() * 1.35)
    _finish(fig, "01_revenue_margin.png",
            "SNACKS and BEBIDAS lead on volume; FRIOS is the margin laggard at 59.6%")


# ---------------------------------------------------------------------------
def chart_02_segments():
    t = load_and_validate_all()
    p = t["pedidos"].merge(t["clientes"][["cliente_id", "segmento"]], on="cliente_id")
    p = p[p["status_pedido"] == "FATURADO"]
    g = p.groupby("segmento").agg(
        receita=("valor_mercadoria_total_brl", "sum"),
        clientes=("cliente_id", "nunique"),
    ).reset_index()
    g["por_cliente"] = g["receita"] / g["clientes"] / 1e3
    g = g.sort_values("por_cliente")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(g))
    bars = ax.barh(y, g["por_cliente"], color=BLUE, alpha=0.85, height=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(g["segmento"], fontsize=10)
    ax.set_xlabel("Revenue per account (R$ thousands)")
    ax.set_title("Not all segments are worth the same effort")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    for bar, (_, row) in zip(bars, g.iterrows()):
        ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height() / 2,
                f"R$ {row.por_cliente:,.0f}k  ·  {int(row.clientes)} accounts",
                va="center", fontsize=9, color=INK)

    ax.set_xlim(0, g["por_cliente"].max() * 1.32)
    _finish(fig, "02_segments.png",
            "A MERCEARIA account is worth 2.4x an ACOUGUEIRO account")


# ---------------------------------------------------------------------------
def chart_03_elasticity():
    t = load_and_validate_all()
    el = build_elasticity_features(t["pedidos"], t["itens_pedido"],
                                   t["produtos"], t["clientes"], with_macro=False)
    import statsmodels.formula.api as smf
    el = el.sample(min(200_000, len(el)), random_state=42)
    el["lp_c"] = el["log_preco"] - el.groupby(["produto_id", "mes"])["log_preco"].transform("mean")
    el["lq_c"] = el["log_qty"] - el.groupby(["produto_id", "mes"])["log_qty"].transform("mean")

    rows = []
    for cat in sorted(el["categoria"].unique()):
        sub = el[el["categoria"] == cat]
        m = smf.ols("lq_c ~ lp_c - 1", data=sub).fit()
        rows.append({"categoria": cat, "eps": -m.params["lp_c"],
                     "p": m.pvalues["lp_c"], "n": len(sub)})
    df = pd.DataFrame(rows).sort_values("eps")

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(df))
    colors = [GREEN if p < 0.05 else GRID for p in df["p"]]
    ax.barh(y, df["eps"], color=colors, height=0.62, zorder=3)
    ax.axvline(0, color=INK, lw=1.2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(df["categoria"], fontsize=10)
    ax.set_xlabel("Estimated elasticity  (negative = demand falls when price rises)")
    ax.set_title("Price response is weak in almost every category")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    for i, (_, row) in enumerate(df.iterrows()):
        label = f"{row.eps:+.3f}" + ("" if row.p < 0.05 else "  (n.s.)")
        ax.text(row.eps - 0.004, i, label, va="center", ha="right",
                fontsize=9, color=INK if row.p < 0.05 else MUTED)

    ax.set_xlim(df["eps"].min() * 1.4, 0.01)
    _finish(fig, "03_elasticity.png",
            "Green = significant at 5%. Only HORTIFRUTI and CONGELADOS show any response")


# ---------------------------------------------------------------------------
def chart_04_churn():
    t = load_and_validate_all()
    ch = build_churn_features(t["clientes"], t["pedidos"], t["itens_pedido"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1]})

    ax = axes[0]
    ax.hist(ch["recencia_dias"], bins=32, color=GREEN, alpha=0.8, zorder=3)
    ax.axvline(ch["recencia_dias"].max(), color=RED, ls="--", lw=1.6, zorder=4)
    ax.text(ch["recencia_dias"].max() - 3, ax.get_ylim()[1] * 0.85,
            f"max = {ch.recencia_dias.max():.0f} days",
            ha="right", fontsize=10, color=RED, fontweight="bold")
    ax.set_xlabel("Days since last order")
    ax.set_ylabel("Customers")
    ax.set_title("Everybody is still buying")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    ax = axes[1]
    order = (ch.groupby("segmento")["variacao_receita_pct"].median()
               .sort_values().index.tolist())
    sns.boxplot(data=ch, x="variacao_receita_pct", y="segmento", order=order,
                color=BLUE, ax=ax, width=0.55, fliersize=2, linewidth=1)
    ax.axvline(-0.5, color=RED, ls="--", lw=1.6)
    ax.text(-0.48, len(order) - 0.6, "churn cut-off (-50%)",
            fontsize=9, color=RED, ha="left")
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("Revenue change, last 6 months vs prior 6")
    ax.set_ylabel("")
    ax.set_title("Variation, not churn")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle("The Atlas has no churners - so churn AUC 0.537 is the correct answer",
                 fontsize=14, fontweight="bold", color=INK, y=1.02)
    _finish(fig, "04_churn.png")


# ---------------------------------------------------------------------------
def chart_05_popularity():
    t = load_and_validate_all()
    inter = build_interactions(t["pedidos"], t["itens_pedido"])
    pop = inter.groupby("produto_id")["n"].sum().sort_values(ascending=False)
    cum = pop.cumsum() / pop.sum() * 100
    n80 = int((cum < 80).sum()) + 1

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(range(1, len(cum) + 1), cum.values, color=GREEN, lw=2.6, zorder=3)
    ax.fill_between(range(1, len(cum) + 1), cum.values, color=GREEN, alpha=0.12, zorder=2)
    ax.axhline(80, color=MUTED, ls=":", lw=1.4)
    ax.axvline(n80, color=RED, ls="--", lw=1.4)
    ax.scatter([n80], [80], color=RED, s=70, zorder=5)
    ax.annotate(f"{n80} SKUs  =  80% of volume",
                xy=(n80, 80), xytext=(n80 + len(cum) * 0.08, 55),
                fontsize=11, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_xlabel("SKUs ranked by units sold")
    ax.set_ylabel("Cumulative % of units")
    ax.set_title("Demand is concentrated - a popularity baseline is hard to beat")
    ax.set_ylim(0, 102)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    _finish(fig, "05_popularity.png")


# ---------------------------------------------------------------------------
def chart_06_ab():
    t = load_and_validate_all()
    out = simulate_experiment(t["pedidos"], t["itens_pedido"], uplift=0.10)
    c, tr = out["control"], out["treatment"]

    rng = np.random.default_rng(42)
    diffs = [(rng.choice(tr, len(tr), True).mean() - rng.choice(c, len(c), True).mean())
             / rng.choice(c, len(c), True).mean() * 100 for _ in range(600)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.hist(c, bins=45, alpha=0.55, color=MUTED, density=True, label="Control", zorder=3)
    ax.hist(tr, bins=45, alpha=0.55, color=GREEN, density=True, label="Treatment", zorder=3)
    ax.axvline(c.mean(), color=MUTED, ls="--", lw=1.8)
    ax.axvline(tr.mean(), color=GREEN, ls="--", lw=1.8)
    ax.set_xlabel("Revenue per order (R$)")
    ax.set_ylabel("Density")
    ax.set_title("The distributions do not overlap")
    ax.legend(loc="upper right")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.hist(diffs, bins=40, color=BLUE, alpha=0.8, zorder=3)
    ax.axvline(0, color=RED, ls="--", lw=1.6, label="No effect")
    ax.axvline(np.mean(diffs), color=INK, lw=2, label=f"Mean = {np.mean(diffs):.1f}%")
    ax.set_xlabel("Uplift (%)")
    ax.set_ylabel("Bootstrap draws")
    ax.set_title("Bootstrap says the effect is real")
    ax.legend(loc="upper left")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle("A/B test harness: a 10% uplift is detected cleanly",
                 fontsize=14, fontweight="bold", color=INK, y=1.02)
    _finish(fig, "06_ab_test.png")


# ---------------------------------------------------------------------------
def chart_07_cepea():
    from src.data.macro import fetch_cepea_daily, monthly_macro
    macro = monthly_macro(fetch_cepea_daily())
    keep = {"cafe": "Coffee", "milho": "Corn", "boi_gordo": "Cattle", "soja": "Soy"}
    macro = macro[macro["commodity"].isin(keep)].copy()
    macro["ano_mes"] = pd.to_datetime(macro["ano_mes"] + "-01")
    macro["idx"] = macro.groupby("commodity")["preco_commodity"].transform(
        lambda s: s / s.iloc[0] * 100)

    fig, ax = plt.subplots(figsize=(12, 5.8))
    colors = {"cafe": "#92400e", "milho": AMBER, "boi_gordo": RED, "soja": GREEN}
    for c, label in keep.items():
        sub = macro[macro["commodity"] == c].sort_values("ano_mes")
        ax.plot(sub["ano_mes"], sub["idx"], lw=2.4, label=label,
                color=colors[c], zorder=3)
        last = sub.iloc[-1]
        ax.annotate(f"{last.idx:.0f}", xy=(last.ano_mes, last.idx),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=10, color=colors[c], fontweight="bold", va="center")

    ax.axhline(100, color=MUTED, ls=":", lw=1.3)
    ax.set_ylabel("Index (first month = 100)")
    ax.set_xlabel("")
    ax.set_title("Real CEPEA indices that feed SKU cost and therefore price")
    ax.legend(loc="upper left", ncol=4)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    _finish(fig, "07_cepea.png")


# ---------------------------------------------------------------------------
def chart_08_benchmark():
    metrics = ["Elasticity\n(R²)", "Churn\n(AUC)", "Recommendation\n(recall@10)",
               "Segmentation\n(silhouette)"]
    atlas = [0.004, 0.537, 0.208, 0.413]
    synth = [0.700, 0.675, 0.244, 0.559]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(metrics))
    w = 0.36
    b1 = ax.bar(x - w/2, atlas, w, label="Atacado Atlas", color="#cbd5e1", zorder=3)
    b2 = ax.bar(x + w/2, synth, w, label="GIRO-SYNTH", color=GREEN, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Same pipeline, two datasets")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 0.82)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    for bars, color in ((b1, MUTED), (b2, INK)):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                    f"{h:.3f}", ha="center", fontsize=9.5,
                    color=color, fontweight="bold")

    _finish(fig, "08_benchmark.png",
            "A known ground truth turns an unlearnable signal into a measurable one")


if __name__ == "__main__":
    print("Generating clean charts...")
    chart_01_revenue_margin()
    chart_02_segments()
    chart_03_elasticity()
    chart_04_churn()
    chart_05_popularity()
    chart_06_ab()
    chart_07_cepea()
    chart_08_benchmark()
    print(f"\nDone -> {OUT}")
