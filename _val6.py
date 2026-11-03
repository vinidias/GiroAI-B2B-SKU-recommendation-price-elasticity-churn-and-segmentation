import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

d = "D:/Dev/giro-ai/giro-synth/data_small"
gt = pd.read_csv(f"{d}/ground_truth/gt_demand.csv")
prod = pd.read_csv(f"{d}/operational/produtos.csv")
df = gt.merge(prod[["produto_id","elasticidade_verdadeira"]], on="produto_id")

# Normalizar por SKU x mes (efeito fixo absorvido)
grp = df.groupby(["produto_id","ano_mes"])
df["log_preco"] = np.log(df["preco_efetivo"])
df["log_qty"] = np.log(df["quantidade"])
df["log_preco_c"] = df["log_preco"] - grp["log_preco"].transform("mean")
df["log_qty_c"] = df["log_qty"] - grp["log_qty"].transform("mean")

# within SKU x mes
m = smf.ols("log_qty_c ~ log_preco_c - 1", data=df).fit()
print(f"beta (within SKU x mes): {m.params['log_preco_c']:.4f}")
print(f"  => elasticidade estimada: {-m.params['log_preco_c']:.4f}")
print(f"  => R2: {m.rsquared:.4f}")
print()

# Por SKU
print("Por SKU (top 10):")
for pid in df.produto_id.unique()[:10]:
    s = df[df.produto_id == pid]
    if s["log_preco_c"].std() < 1e-6:
        continue
    mm = smf.ols("log_qty_c ~ log_preco_c - 1", data=s).fit()
    print(f"  {pid}: beta={mm.params['log_preco_c']:.3f}, true_eps={s.elasticidade_verdadeira.iloc[0]:.2f}")