import pandas as pd
from pathlib import Path

RAW = Path("D:/Dev/giro-ai/data/raw")
man = pd.read_csv(RAW/"manifesto.csv")
print("manifesto:")
print(man.to_string(index=False))
print()

# Contar raw e apos dedup simples
raw = pd.read_csv(RAW/"itens_pedido.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
print(f"raw itens_pedido: {len(raw):,}")
print(f"apos drop_duplicates(): {len(raw.drop_duplicates()):,}")
print(f"apos drop_duplicates(item_pedido_id): {len(raw.drop_duplicates(subset=['item_pedido_id'])):,}")

# IDs duplicados
g = raw.groupby("item_pedido_id").size()
dup_ids = g[g > 1]
print(f"\nIDs duplicados: {len(dup_ids)}")
print(dup_ids.to_string())

# Verificar se há ID vazio
print(f"\nIDs vazios: {(raw.item_pedido_id == '').sum()}")
empty = raw[raw.item_pedido_id == ""]
print(f"linhas com ID vazio: {len(empty)}")
if len(empty):
    print(empty.head(3).to_string())