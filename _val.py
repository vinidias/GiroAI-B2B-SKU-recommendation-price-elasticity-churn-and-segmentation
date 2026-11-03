import pandas as pd
import numpy as np

d = "D:/Dev/giro-ai/giro-synth/data_small"
gt_el = pd.read_csv(f"{d}/ground_truth/gt_elasticity.csv")
itens = pd.read_csv(f"{d}/operational/itens_pedido.csv", dtype={"preco_unitario_brl": str})
pedidos = pd.read_csv(f"{d}/operational/pedidos.csv")
produtos = pd.read_csv(f"{d}/operational/produtos.csv")

itens["preco_unitario_brl"] = pd.to_numeric(itens["preco_unitario_brl"], errors="coerce")
itens = itens.dropna(subset=["preco_unitario_brl"])
df = itens.merge(pedidos[["pedido_id","data_pedido","status_pedido"]], on="pedido_id")
df = df.merge(produtos[["produto_id","categoria"]], on="produto_id")
df["log_qty"] = np.log1p(df["quantidade_expedida"])
df["log_preco"] = np.log(df["preco_unitario_brl"].clip(lower=1e-6))
print("obs:", len(df), "| qty distribuicao:", df.quantidade_expedida.describe().to_dict())
print()
import statsmodels.formula.api as smf
m = smf.ols("log_qty ~ log_preco * C(categoria)", data=df).fit()
print("R2:", round(m.rsquared, 4))
print()
print("Verdade vs estimado por categoria:")
for _, row in gt_el.iterrows():
    cat = row["categoria"]
    key = f"log_preco:C(categoria)[T.{cat}]"
    base = m.params.get("log_preco", 0)
    est = base + m.params.get(key, 0) if cat != sorted(df.categoria.unique())[0] else base
    print(f"  {cat:14s} true={row['epsilon']:.2f} est={-est:.2f}")
print()
gt_ch = pd.read_csv(f"{d}/ground_truth/gt_churn.csv")
print("churn rate:", round((gt_ch.churn_month_index >= 0).mean(), 3), f"({(gt_ch.churn_month_index >= 0).sum()}/{len(gt_ch)})")