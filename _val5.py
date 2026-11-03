import pandas as pd
import numpy as np

d = "D:/Dev/giro-ai/giro-synth/data_small"
gt = pd.read_csv(f"{d}/ground_truth/gt_demand.csv")
prod = pd.read_csv(f"{d}/operational/produtos.csv")
itens = pd.read_csv(f"{d}/operational/itens_pedido.csv", dtype={"preco_unitario_brl":str})
itens["preco_unitario_brl"] = pd.to_numeric(itens["preco_unitario_brl"], errors="coerce")
itens = itens.dropna(subset=["preco_unitario_brl"])

# Comparar GT demand com itens reais
print("gt_demand cols:", gt.columns.tolist())
print("itens cols:", itens.columns.tolist())
print()
print("gt_demand sample:")
print(gt.head(5).to_string(index=False))
print()
print("itens sample:")
print(itens.head(5).to_string(index=False))
print()
# O GT demand preco_efetivo é o preco com promo, e o itens tem preco_unitario + desconto
# Verificar se batem
print("gt_demand preco range:", gt.preco_efetivo.min(), "-", gt.preco_efetivo.max())
print("itens preco range:", itens.preco_unitario_brl.min(), "-", itens.preco_unitario_brl.max())