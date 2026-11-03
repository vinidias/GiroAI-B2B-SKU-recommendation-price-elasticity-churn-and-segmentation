import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

d = "D:/Dev/giro-ai/giro-synth/data_small"
gt_el = pd.read_csv(f"{d}/ground_truth/gt_elasticity.csv")
gt_demand = pd.read_csv(f"{d}/ground_truth/gt_demand.csv")
produtos = pd.read_csv(f"{d}/operational/produtos.csv")

df = gt_demand.merge(produtos[["produto_id","categoria","elasticidade_verdadeira","preco_lista_brl"]], on="produto_id")
df["log_qty"] = np.log(df["quantidade"].clip(lower=1))
df["log_preco"] = np.log(df["preco_efetivo"].clip(lower=1e-6))

print("obs:", len(df))
print("qty range:", df.quantidade.min(), "-", df.quantidade.max())
print("preco range:", round(df.preco_efetivo.min(),2), "-", round(df.preco_efetivo.max(),2))
print("SKUs distintos:", df.produto_id.nunique())
print("observacoes por SKU (media):", round(len(df)/df.produto_id.nunique(), 1))
print()

# Modelo com efeito fixo por produto (within estimator)
# log_qty = alpha_i + beta * log_preco  (beta = -epsilon por SKU)
# Simplificado: OLS com dummies de produto
import statsmodels.api as sm
X = pd.get_dummies(df["produto_id"], drop_first=True, dtype=float)
X["log_preco"] = df["log_preco"].values
X = sm.add_constant(X)
y = df["log_qty"].values
model = sm.OLS(y, X).fit()
beta = model.params["log_preco"]
print(f"Elasticidade within-SKU (pooled): {-beta:.3f}")
print(f"R2: {model.rsquared:.4f}")
print()
print("Verdade por SKU (amostra):")
print(produtos[["produto_id","categoria","elasticidade_verdadeira"]].head(10).to_string(index=False))