import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("D:/Dev/giro-ai/data/raw")
def load(n): return pd.read_csv(RAW / n, encoding="utf-8-sig", dtype=str, keep_default_na=False)

print("=" * 70)
print("A. INTEGRIDADE REFERENCIAL (FKs orfas)")
print("=" * 70)

def check_fk(child, child_col, parent, parent_col):
    c = load(f"{child}.csv"); p = load(f"{parent}.csv")
    child_vals = set(c[child_col]) - {""}
    parent_vals = set(p[parent_col]) - {""}
    orphans = child_vals - parent_vals
    if orphans:
        print(f"  {child}.{child_col} -> {parent}.{parent_col}: {len(orphans)} orfaos: {list(orphans)[:5]}")

check_fk("pedidos", "cliente_id", "clientes", "cliente_id")
check_fk("pedidos", "vendedor_id", "vendedores", "vendedor_id")
check_fk("pedidos", "tabela_preco_aplicada", "tabelas_preco", "tabela_preco_id")
check_fk("itens_pedido", "pedido_id", "pedidos", "pedido_id")
check_fk("itens_pedido", "produto_id", "produtos", "produto_id")
check_fk("clientes", "vendedor_id", "vendedores", "vendedor_id")
check_fk("clientes", "tabela_preco_id", "tabelas_preco", "tabela_preco_id")
check_fk("precos_produto", "produto_id", "produtos", "produto_id")
check_fk("precos_produto", "tabela_preco_id", "tabelas_preco", "tabela_preco_id")
check_fk("produtos", "fornecedor_id", "fornecedores", "fornecedor_id")
check_fk("lotes", "produto_id", "produtos", "produto_id")
check_fk("movimentos_estoque", "lote_id", "lotes", "lote_id")
check_fk("titulos_financeiros", "pedido_id", "pedidos", "pedido_id")
check_fk("baixas_financeiras", "titulo_id", "titulos_financeiros", "titulo_id")
check_fk("devolucoes", "pedido_id", "pedidos", "pedido_id")
check_fk("devolucoes", "produto_id", "produtos", "produto_id")
check_fk("paradas", "pedido_id", "pedidos", "pedido_id")
check_fk("romaneios", "rota_id", "rotas", "rota_id")
check_fk("compras", "fornecedor_id", "fornecedores", "fornecedor_id")
check_fk("itens_compra", "compra_id", "compras", "compra_id")
check_fk("itens_compra", "produto_id", "produtos", "produto_id")
print("  (fim)")