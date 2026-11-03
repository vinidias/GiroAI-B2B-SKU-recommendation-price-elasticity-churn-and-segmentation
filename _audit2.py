import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("D:/Dev/giro-ai/data/raw")

def load(name):
    return pd.read_csv(RAW / name, encoding="utf-8-sig", dtype=str, keep_default_na=False)

print("=" * 70)
print("1. NULOS / STRINGS VAZIAS POR COLUNA")
print("=" * 70)
for name in ["clientes","produtos","pedidos","itens_pedido","precos_produto",
             "tabelas_preco","vendedores","lotes","movimentos_estoque",
             "titulos_financeiros","baixas_financeiras","devolucoes",
             "estoque_diario","compras","itens_compra","fornecedores",
             "romaneios","paradas","rotas","veiculos","limite_credito",
             "ocorrencias_recebimento"]:
    df = load(f"{name}.csv")
    empties = (df == "").sum()
    total = len(df)
    cols_with_empty = empties[empties > 0]
    if len(cols_with_empty):
        print(f"\n{name} ({total:,} linhas):")
        for col, n in cols_with_empty.items():
            print(f"  {col:35s} {n:>7,} ({n/total*100:.1f}%)")