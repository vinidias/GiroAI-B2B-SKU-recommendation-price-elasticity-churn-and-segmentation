"""Fallback synthetic generator mimicking the Atacado Atlas layout."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_DIR, RANDOM_STATE, RAW_FILES

CATEGORIES = ["Bebidas", "Mercearia", "Limpeza", "Perecíveis", "Congelados", "Higiene"]
REGIONS = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]
CHANNELS = ["Horeca", "Retail"]


def generate(n_customers: int = 400, n_products: int = 120, n_orders: int = 6000,
             days: int = 730, seed: int = RANDOM_STATE) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01")

    clientes = pd.DataFrame({
        "cliente_id": np.arange(1, n_customers + 1),
        "nome": [f"Cliente_{i:04d}" for i in range(1, n_customers + 1)],
        "regiao": rng.choice(REGIONS, n_customers),
        "canal": rng.choice(CHANNELS, n_customers, p=[0.6, 0.4]),
        "data_cadastro": start - pd.to_timedelta(rng.integers(30, 900, n_customers), unit="D"),
    })

    n_suppliers = 25
    produtos = pd.DataFrame({
        "produto_id": np.arange(1, n_products + 1),
        "nome": [f"SKU_{i:04d}" for i in range(1, n_products + 1)],
        "categoria": rng.choice(CATEGORIES, n_products),
        "fornecedor_id": rng.integers(1, n_suppliers + 1, n_products),
        "custo_unitario": np.round(rng.uniform(2, 60, n_products), 2),
    })

    base_price = (produtos["custo_unitario"] * rng.uniform(1.2, 2.0, n_products)).round(2)
    price_rows = []
    for pid, bp in zip(produtos["produto_id"], base_price, strict=False):
        for _ in range(rng.integers(3, 7)):
            mult = rng.uniform(0.85, 1.15)
            price_rows.append({
                "produto_id": int(pid),
                "data_vigencia": start + pd.to_timedelta(int(rng.integers(0, days)), unit="D"),
                "preco": round(float(bp * mult), 2),
            })
    precos_produto = pd.DataFrame(price_rows).sort_values(["produto_id", "data_vigencia"]).reset_index(drop=True)

    vendedores = pd.DataFrame({
        "vendedor_id": np.arange(1, 16),
        "nome": [f"Vendedor_{i:02d}" for i in range(1, 16)],
        "regiao": rng.choice(REGIONS, 15),
    })

    pop = rng.pareto(1.5, n_products) + 1
    pop = pop / pop.sum()
    cust_w = rng.pareto(1.8, n_customers) + 1
    cust_w = cust_w / cust_w.sum()

    pedidos = pd.DataFrame({
        "pedido_id": np.arange(1, n_orders + 1),
        "cliente_id": rng.choice(clientes["cliente_id"], n_orders, p=cust_w),
        "vendedor_id": rng.integers(1, 16, n_orders),
        "data_pedido": start + pd.to_timedelta(rng.integers(0, days, n_orders), unit="D"),
        "status": rng.choice(["entregue", "cancelado", "devolvido"], n_orders, p=[0.9, 0.05, 0.05]),
    })

    item_rows = []
    for oid, odate in zip(pedidos["pedido_id"], pedidos["data_pedido"], strict=False):
        n_items = int(rng.integers(1, 7))
        chosen = rng.choice(produtos["produto_id"], size=n_items, replace=False, p=pop)
        for pid in chosen:
            pp = precos_produto[(precos_produto["produto_id"] == pid) & (precos_produto["data_vigencia"] <= odate)]
            price = float(pp["preco"].iloc[-1]) if len(pp) else float(base_price[int(pid) - 1])
            # elastic demand: quantity decreases with price relative to base price
            base = float(base_price[int(pid) - 1])
            price_ratio = max(price, 0.01) / max(base, 0.01)
            # strong, clean elasticity signal (elastic demand, eps = -1.5)
            expected_qty = 30 * (price_ratio ** -1.5)
            qty = max(1, int(round(expected_qty * rng.lognormal(0, 0.15))))
            item_rows.append({
                "pedido_id": int(oid),
                "produto_id": int(pid),
                "quantidade": qty,
                "preco_unitario": round(price, 2),
                "desconto_pct": round(float(rng.uniform(0, 0.15)), 4),
            })
    itens_pedido = pd.DataFrame(item_rows)
    itens_pedido["valor_total"] = (itens_pedido["quantidade"] * itens_pedido["preco_unitario"]
                                   * (1 - itens_pedido["desconto_pct"])).round(2)

    cutoff = pedidos["data_pedido"].max() - pd.Timedelta(days=90)
    recent = pedidos[pedidos["data_pedido"] >= cutoff].groupby("cliente_id").size()
    churn = pd.DataFrame({"cliente_id": clientes["cliente_id"]})
    churn["compras_90d"] = churn["cliente_id"].map(recent).fillna(0).astype(int)
    churn["churn"] = (churn["compras_90d"] == 0).astype(int)
    clientes = clientes.merge(churn[["cliente_id", "churn"]], on="cliente_id", how="left")

    return {
        "clientes": clientes,
        "produtos": produtos,
        "pedidos": pedidos,
        "itens_pedido": itens_pedido,
        "precos_produto": precos_produto,
        "vendedores": vendedores,
    }


def write_to_disk(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(DATA_DIR / RAW_FILES[name], index=False)


if __name__ == "__main__":
    tables = generate()
    write_to_disk(tables)
    for name, df in tables.items():
        print(f"{name:15s} rows={len(df):6d} cols={len(df.columns)}")

