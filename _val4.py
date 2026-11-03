import pandas as pd
import numpy as np

d = "D:/Dev/giro-ai/giro-synth/data_small"
gt = pd.read_csv(f"{d}/ground_truth/gt_demand.csv")
prod = pd.read_csv(f"{d}/operational/produtos.csv")
df = gt.merge(prod[["produto_id","elasticidade_verdadeira"]], on="produto_id")
df["log_qty"] = np.log(df["quantidade"])
df["log_preco"] = np.log(df["preco_efetivo"])

# Verificar: para o mesmo SKU e mesmo mes, qty cai quando preco cai? (deveria SUBIR)
sku = "P0005"
sub = df[(df.produto_id == sku) & (df.ano_mes == "2024-09")].sort_values("preco_efetivo")
print(f"SKU {sku} mes 2024-09, eps_true=1.56:")
print(sub[["preco_efetivo","quantidade"]].drop_duplicates().to_string(index=False))
print()
# Regressao WITHIN SKU x MES
df["preco_norm"] = df["preco_efetivo"] / df.groupby(["produto_id","ano_mes"])["preco_efetivo"].transform("mean")
df["qty_norm"] = df["quantidade"] / df.groupby(["produto_id","ano_mes"])["quantidade"].transform("mean")
mask = (df.preco_norm != 1.0) & df.preco_norm.notna()
s = df[mask]
c = s["preco_norm"].corr(s["qty_norm"])
print(f"Correlacao within SKU x mes: {c:.3f} (esperado negativo)")
print()
# Regressao simples com FE
import statsmodels.formula.api as smf
m = smf.ols("log_qty ~ log_preco + C(produto_id) + C(ano_mes)", data=df).fit()
print(f"beta (log_preco): {m.params['log_preco']:.3f}  => eps={-m.params['log_preco']:.3f}")
print(f"R2: {m.rsquared:.4f}")