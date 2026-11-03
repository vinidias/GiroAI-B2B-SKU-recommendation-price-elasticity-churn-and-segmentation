import pandas as pd
import numpy as np

d = "D:/Dev/giro-ai/giro-synth/data_small"
gt_demand = pd.read_csv(f"{d}/ground_truth/gt_demand.csv")
produtos = pd.read_csv(f"{d}/operational/produtos.csv")

df = gt_demand.merge(produtos[["produto_id","categoria","elasticidade_verdadeira","preco_lista_brl"]], on="produto_id")
df["log_qty"] = np.log(df["quantidade"].clip(lower=1))
df["log_preco"] = np.log(df["preco_efetivo"].clip(lower=1e-6))

# Um SKU específico
sku = "P0005"
sub = df[df.produto_id == sku].sort_values("preco_efetivo")
print(f"SKU {sku} (true eps=1.5587), {len(sub)} obs")
print(sub[["ano_mes","preco_efetivo","quantidade"]].head(15).to_string(index=False))
print()
print("Correlacao preco-qty por SKU (primeiros 10):")
for pid in df.produto_id.unique()[:10]:
    s = df[df.produto_id == pid]
    c = s["log_preco"].corr(s["log_qty"])
    print(f"  {pid}: corr={c:.3f}, eps_true={s.elasticidade_verdadeira.iloc[0]:.2f}")