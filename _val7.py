import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

d = "D:/Dev/giro-ai/giro-synth/data_small"
gt = pd.read_csv(f"{d}/ground_truth/gt_demand.csv")
prod = pd.read_csv(f"{d}/operational/produtos.csv")
df = gt.merge(prod[["produto_id","elasticidade_verdadeira"]], on="produto_id")

# Within SKU x mes usando o LOG do preco relativo a media do SKU x mes
grp = df.groupby(["produto_id","ano_mes"])
df["log_preco"] = np.log(df["preco_efetivo"])
df["log_qty"] = np.log(df["quantidade"])
df["log_preco_c"] = df["log_preco"] - grp["log_preco"].transform("mean")
df["log_qty_c"] = df["log_qty"] - grp["log_qty"].transform("mean")

m = smf.ols("log_qty_c ~ log_preco_c - 1", data=df).fit()
print(f"Pooled within-SKU: eps={-m.params['log_preco_c']:.3f}, R2={m.rsquared:.3f}")
print()

# Por SKU: medir correlacao entre verdade e estimado
truths, ests = [], []
for pid in df.produto_id.unique():
    s = df[df.produto_id == pid]
    if s["log_preco_c"].std() < 1e-6:
        continue
    mm = smf.ols("log_qty_c ~ log_preco_c - 1", data=s).fit()
    truths.append(s.elasticidade_verdadeira.iloc[0])
    ests.append(-mm.params["log_preco_c"])

from scipy.stats import pearsonr, spearmanr
print(f"Correlacao verdade x estimado (Pearson):  {pearsonr(truths, ests)[0]:.3f}")
print(f"Correlacao verdade x estimado (Spearman): {spearmanr(truths, ests)[0]:.3f}")
print(f"MAE: {np.mean(np.abs(np.array(truths) - np.array(ests))):.3f}")
print(f"Vies medio: {np.mean(np.array(ests) - np.array(truths)):.3f}")