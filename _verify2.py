import pandas as pd
from src.data.validate import load_raw, load_and_validate_all

raw = load_raw("itens_pedido")
print(f"load_raw: {len(raw):,} (esperado 147,053)")

t = load_and_validate_all()
it = t["itens_pedido"]
print(f"load_and_validate_all: {len(it):,}")

# O que sobra?
print(f"P999 em raw: {(raw.produto_id == 'P999').sum()}")
# Verificar se algum item foi dropado
merged = raw.merge(t["produtos"][["produto_id","categoria"]], on="produto_id", how="left")
orphans = merged[merged.categoria.isna()]
print(f"itens com produto inexistente: {len(orphans)}")
print(orphans[["item_pedido_id","produto_id"]].to_string(index=False) if len(orphans) else "nenhum")

# Verificar duplicatas por item_pedido_id
print(f"item_pedido_id unicos: {it.item_pedido_id.nunique():,}")
print(f"linhas: {len(it):,}")
dups = it[it.item_pedido_id.duplicated(keep=False)]
print(f"duplicados: {len(dups)}")
if len(dups):
    print(dups[["item_pedido_id","pedido_id","produto_id"]].to_string(index=False))