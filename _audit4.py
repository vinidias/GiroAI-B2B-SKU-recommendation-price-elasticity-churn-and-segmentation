import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("D:/Dev/giro-ai/data/raw")
def load(n): return pd.read_csv(RAW / n, encoding="utf-8-sig", dtype=str, keep_default_na=False)

print("=" * 70)
print("B. VALORES NAO-NUMERICOS EM COLUNAS NUMERICAS")
print("=" * 70)
num_cols = {
    "pedidos": ["valor_mercadoria_total_brl","peso_total_solicitado_kg","frete_rateado_brl","limite_credito_aplicado_brl","exposicao_credito_data_brl"],
    "itens_pedido": ["quantidade_solicitada","quantidade_expedida","preco_unitario_brl","desconto_pct","valor_mercadoria_brl","tributos_brl","comissao_brl","custo_unitario_estoque_medio_brl","cmv_brl"],
    "produtos": ["custo_referencia_brl","preco_lista_brl","peso_unitario_kg","fator_conversao"],
    "clientes": ["limite_credito_brl","prazo_pagamento_dias"],
    "lotes": ["custo_unitario_brl"],
    "movimentos_estoque": ["quantidade_un","valor_custo_brl"],
    "titulos_financeiros": ["valor_original_brl"],
    "baixas_financeiras": ["valor_baixado_brl","valor_juros_multa_brl"],
    "devolucoes": ["quantidade_un","valor_creditado_brl"],
    "estoque_diario": ["quantidade_estoque_un","valor_estoque_brl","quantidade_vencendo_7d_un"],
    "compras": [],
    "itens_compra": ["quantidade_pedida_un","quantidade_recebida_un","custo_unitario_brl","custo_unitario_efetivo_brl","bonificacao_pct"],
    "paradas": ["quantidade_entregue_un","quantidade_recusada_un","km_acumulado"],
    "romaneios": ["km_total","custo_km_brl","custo_romaneio_brl","pedidos_planejados","paradas_realizadas"],
    "precos_produto": ["preco_brl"],
    "tabelas_preco": ["fator_sobre_lista"],
    "vendedores": ["comissao_pct","meta_mensal_brl"],
    "limite_credito": ["limite_brl"],
    "ocorrencias_recebimento": ["quantidade_afetada_un","valor_credito_brl"],
}
for name, cols in num_cols.items():
    df = load(f"{name}.csv")
    for c in cols:
        if c not in df.columns: continue
        bad = pd.to_numeric(df[c], errors="coerce").isna() & (df[c] != "")
        if bad.any():
            samples = df.loc[bad, c].unique()[:5]
            print(f"  {name}.{c}: {bad.sum()} invalidos -> {list(samples)}")

print()
print("=" * 70)
print("C. REGRAS DE NEGOCIO VIOLADAS")
print("=" * 70)

# 1. Quantidade expedida > solicitada
it = load("itens_pedido.csv")
q_sol = pd.to_numeric(it.quantidade_solicitada, errors="coerce")
q_exp = pd.to_numeric(it.quantidade_expedida, errors="coerce")
print(f"  itens: expedida > solicitada: {(q_exp > q_sol).sum()}")

# 2. Preco negativo ou zero
p = pd.to_numeric(it.preco_unitario_brl, errors="coerce")
print(f"  itens: preco <= 0: {(p <= 0).sum()} | preco negativo: {(p < 0).sum()}")

# 3. Desconto fora de [0, 1]
d = pd.to_numeric(it.desconto_pct, errors="coerce")
print(f"  itens: desconto fora de [0,1]: {((d < 0) | (d > 1)).sum()} | max: {d.max()}")

# 4. Valor mercadoria != qty * preco * (1-desc)
it["calc"] = (q_exp * p * (1 - d)).round(2)
it["given"] = pd.to_numeric(it.valor_mercadoria_brl, errors="coerce")
diff = (it.calc - it.given).abs()
print(f"  itens: valor_mercadoria diverge do calculado: {(diff > 0.01).sum()} (max diff R$ {diff.max():.2f})")

# 5. Data pedido no futuro ou anterior ao início
ped = load("pedidos.csv")
ped["data_pedido"] = pd.to_datetime(ped.data_pedido, errors="coerce")
print(f"  pedidos: data invalida: {ped.data_pedido.isna().sum()}")
print(f"  pedidos: fora do periodo 2024-09 a 2026-08: {((ped.data_pedido < '2024-09-01') | (ped.data_pedido > '2026-08-31')).sum()}")

# 6. data_programacao < data_pedido
ped["data_prog"] = pd.to_datetime(ped.data_programacao, errors="coerce")
print(f"  pedidos: programacao antes do pedido: {(ped.data_prog < ped.data_pedido).sum()}")

# 7. Estoque negativo
est = load("estoque_diario.csv")
q = pd.to_numeric(est.quantidade_estoque_un, errors="coerce")
print(f"  estoque: quantidade negativa: {(q < 0).sum()}")

# 8. Data validade < data entrada (lote)
lot = load("lotes.csv")
de = pd.to_datetime(lot.data_entrada, errors="coerce")
dv = pd.to_datetime(lot.data_validade, errors="coerce")
print(f"  lotes: validade antes da entrada: {(dv < de).sum()}")

# 9. Titulo vencimento < emissao
tit = load("titulos_financeiros.csv")
te = pd.to_datetime(tit.data_emissao, errors="coerce")
tv = pd.to_datetime(tit.data_vencimento, errors="coerce")
print(f"  titulos: vencimento antes da emissao: {(tv < te).sum()}")

# 10. Parada com hora_chegada mas sem km
par = load("paradas.csv")
print(f"  paradas: entregue mas sem hora_chegada: {((par.status_parada == 'ENTREGUE') & (par.hora_chegada == '')).sum()}")