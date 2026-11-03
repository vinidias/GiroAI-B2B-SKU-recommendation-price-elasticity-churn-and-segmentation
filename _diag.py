from src.data.validate import load_and_validate_all
import pandas as pd

t = load_and_validate_all()
it = t["itens_pedido"].merge(t["pedidos"][["pedido_id","status_pedido"]], on="pedido_id")
it = it[it.status_pedido=="FATURADO"]
it = it.merge(t["produtos"][["produto_id","categoria","custo_referencia_brl"]], on="produto_id")
it["receita"] = it.quantidade_expedida * it.preco_unitario_brl * (1 - it.desconto_pct)
it["margem"] = it.receita - it.quantidade_expedida * it.custo_referencia_brl

g = it.groupby("categoria").agg(
    receita=("receita","sum"), margem=("margem","sum")
).reset_index()
g["receita_M"] = (g.receita/1e6).round(1)
g["margem_pct"] = (g.margem/g.receita*100).round(1)
print("Receita por categoria (R$ M) - APOS LIMPEZA:")
print(g.sort_values("receita", ascending=False)[["categoria","receita_M","margem_pct"]].to_string(index=False))
print()
print(f"Receita total: R$ {it.receita.sum()/1e6:.1f}M")

# markup por item
it2 = it.copy()
it2["markup"] = it2.preco_unitario_brl / it2.custo_referencia_brl
print(f"Itens com markup > 50x: {(it2.markup > 50).sum():,}")