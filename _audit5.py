import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("D:/Dev/giro-ai/data/raw")
def load(n): return pd.read_csv(RAW / n, encoding="utf-8-sig", dtype=str, keep_default_na=False)

print("=" * 70)
print("ANOMALIA 8: preco com virgula decimal (formato PT-BR)")
print("=" * 70)
it = load("itens_pedido.csv")
mask = it.preco_unitario_brl.str.contains(",", na=False)
print(it.loc[mask, ["item_pedido_id","pedido_id","produto_id","quantidade_solicitada","preco_unitario_brl","valor_mercadoria_brl"]].to_string(index=False))

print()
print("=" * 70)
print("ANOMALIA 9: valor_mercadoria nao bate com qty*preco*(1-desc)")
print("=" * 70)
it = load("itens_pedido.csv")
it["q"] = pd.to_numeric(it.quantidade_expedida, errors="coerce")
it["p"] = pd.to_numeric(it.preco_unitario_brl.str.replace(",", "."), errors="coerce")
it["d"] = pd.to_numeric(it.desconto_pct, errors="coerce")
it["calc"] = (it.q * it.p * (1 - it.d/100)).round(2)
it["given"] = pd.to_numeric(it.valor_mercadoria_brl, errors="coerce")
it["diff"] = (it.calc - it.given).abs()
bad = it[it["diff"] > 0.01]
print(f"Total divergentes: {len(bad):,}")
print(bad.nlargest(10, "diff")[["item_pedido_id","quantidade_expedida","preco_unitario_brl","desconto_pct","valor_mercadoria_brl","calc","diff"]].to_string(index=False))

print()
print("=" * 70)
print("ANOMALIA 10: quantidade_expedida > solicitada")
print("=" * 70)
mask = it["q"] > pd.to_numeric(it.quantidade_solicitada, errors="coerce")
print(it.loc[mask, ["item_pedido_id","pedido_id","quantidade_solicitada","quantidade_expedida"]].to_string(index=False))

print()
print("=" * 70)
print("ANOMALIA 11: datas invalidas em pedidos")
print("=" * 70)
ped = load("pedidos.csv")
bad_date = pd.to_datetime(ped.data_pedido, errors="coerce").isna() & (ped.data_pedido != "")
print(ped.loc[bad_date, ["pedido_id","cliente_id","data_pedido","data_programacao","valor_mercadoria_total_brl"]].to_string(index=False))

print()
print("=" * 70)
print("ANOMALIA 12: item com produto P999 (inexistente)")
print("=" * 70)
print(it[it.produto_id == "P999"][["item_pedido_id","pedido_id","produto_id","quantidade_solicitada","preco_unitario_brl"]].to_string(index=False))