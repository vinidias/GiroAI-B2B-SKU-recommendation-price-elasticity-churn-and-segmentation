import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
from src.data.validate import load_and_validate_all
import pandas as pd

t = load_and_validate_all()
print()
print("=== TABELAS LIMPAS ===")
for k, v in t.items():
    print(f"{k:16s} {v.shape}")

print()
print("=== VALIDACAO POS-LIMPEZA ===")
it = t["itens_pedido"]
print(f"desconto range: {it.desconto_pct.min():.4f} - {it.desconto_pct.max():.4f}")
print(f"qty solicitada negativa: {(it.quantidade_solicitada < 0).sum()}")
print(f"produto P999 presente: {(it.produto_id == 'P999').sum()}")
print(f"preco com virgula (NaN): {it.preco_unitario_brl.isna().sum()}")

# consistencia valor
calc = (it.quantidade_expedida * it.preco_unitario_brl * (1 - it.desconto_pct)).round(2)
mismatch = (it.valor_mercadoria_brl - calc).abs() > 0.01
print(f"valor_mercadoria inconsistente: {mismatch.sum()}")

# markup
it2 = it.merge(t["produtos"][["produto_id","categoria","custo_referencia_brl"]], on="produto_id")
it2["markup"] = it2.preco_unitario_brl / it2.custo_referencia_brl
print(f"markup > 50x: {(it2.markup > 50).sum()}")

# datas
ped = t["pedidos"]
print(f"data_pedido NaN: {ped.data_pedido.isna().sum()}")
print(f"data_pedido range: {ped.data_pedido.min().date()} a {ped.data_pedido.max().date()}")