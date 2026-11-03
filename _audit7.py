import pandas as pd
from pathlib import Path

RAW = Path("D:/Dev/giro-ai/data/raw")
it = pd.read_csv(RAW/"itens_pedido.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
it["p"] = pd.to_numeric(it.preco_unitario_brl.str.replace(",", "."), errors="coerce")
it["q"] = pd.to_numeric(it.quantidade_expedida, errors="coerce")
it["d"] = pd.to_numeric(it.desconto_pct, errors="coerce")
it["v"] = pd.to_numeric(it.valor_mercadoria_brl, errors="coerce")
it["expected_sem"] = (it.q * it.p).round(2)
it["expected_com"] = (it.q * it.p * (1 - it.d/100)).round(2)

dz = it[it.d == 0]
print(f"itens com desconto=0: {len(dz):,}")
print(f"  valor == qty*preco: {((dz.v - dz.expected_sem).abs() <= 0.01).sum():,}")
print(f"  valor != qty*preco: {((dz.v - dz.expected_sem).abs() > 0.01).sum():,}")
print()
dpos = it[it.d > 0]
print(f"itens com desconto>0: {len(dpos):,}")
print(f"  valor == qty*preco*(1-d): {((dpos.v - dpos.expected_com).abs() <= 0.01).sum():,}")
print(f"  valor == qty*preco (desc ignorado): {((dpos.v - dpos.expected_sem).abs() <= 0.01).sum():,}")
print()
print("Conclusao: o Atlas tem DOIS grupos de itens:")
print("  A) valor = qty*preco*(1-desconto)  -> calculo correto")
print("  B) valor = qty*preco              -> desconto exibido mas NAO aplicado")
print()
print("Amostra grupo B (desconto>0 mas valor=qty*preco):")
mask = (dpos.d > 0) & ((dpos.v - dpos.expected_sem).abs() <= 0.01)
print(dpos[mask][["item_pedido_id","q","p","d","v","expected_com","expected_sem"]].head(10).to_string(index=False))
print()
print(f"Total grupo A: {((dpos.v - dpos.expected_com).abs() <= 0.01).sum():,}")
print(f"Total grupo B: {mask.sum():,}")