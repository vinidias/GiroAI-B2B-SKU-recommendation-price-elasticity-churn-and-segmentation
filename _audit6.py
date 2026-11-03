import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("D:/Dev/giro-ai/data/raw")
def load(n): return pd.read_csv(RAW / n, encoding="utf-8-sig", dtype=str, keep_default_na=False)

it = load("itens_pedido.csv")
print("=== ANOMALIA 8: virgula decimal ===")
mask = it.preco_unitario_brl.str.contains(",", na=False)
print(f"total: {mask.sum()} itens")
sub = it[mask]
print(f"pedidos afetados: {sub.pedido_id.nunique()}")
print(f"valores unicos: {sub.preco_unitario_brl.unique()[:20]}")
print()

print("=== ANOMALIA 9: desconto aplicado 2x ===")
it["p"] = pd.to_numeric(it.preco_unitario_brl.str.replace(",", "."), errors="coerce")
it["q"] = pd.to_numeric(it.quantidade_expedida, errors="coerce")
it["d"] = pd.to_numeric(it.desconto_pct, errors="coerce")
it["v"] = pd.to_numeric(it.valor_mercadoria_brl, errors="coerce")
it["com_desc"] = (it.q * it.p * (1 - it.d/100)).round(2)
it["sem_desc"] = (it.q * it.p).round(2)
d1 = (it.v - it.com_desc).abs()
d2 = (it.v - it.sem_desc).abs()
print(f"valor bate COM desconto: {(d1 <= 0.01).sum():,}")
print(f"valor bate SEM desconto: {(d2 <= 0.01).sum():,}")
print(f"nao bate com nenhum: {((d1 > 0.01) & (d2 > 0.01)).sum():,}")
print()
print("Exemplos em que valor = qty*preco (desconto ignorado):")
mask = (d2 <= 0.01) & (it.d > 0)
print(it.loc[mask, ["item_pedido_id","q","p","d","v","sem_desc"]].head(5).to_string(index=False))

print()
print("=== ANOMALIA 10: qty solicitada negativa ===")
it["qs"] = pd.to_numeric(it.quantidade_solicitada, errors="coerce")
print(it[it.qs < 0][["item_pedido_id","pedido_id","quantidade_solicitada","quantidade_expedida"]].to_string(index=False))

print()
print("=== ANOMALIA 11: formato de data misto ===")
ped = load("pedidos.csv")
print(f"datas no formato DD/MM/YYYY: {ped.data_pedido.str.contains('/').sum()}")
print(f"datas no formato YYYY-MM-DD: {ped.data_pedido.str.contains('-').sum()}")
print()
print("Amostra dos dois formatos:")
print("DD/MM:", ped[ped.data_pedido.str.contains('/')].pedido_id.head(3).tolist())
print("YYYY-MM:", ped[ped.data_pedido.str.contains('-')].pedido_id.head(3).tolist())