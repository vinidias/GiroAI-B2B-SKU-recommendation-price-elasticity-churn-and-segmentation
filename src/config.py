"""Central configuration for Giro AI."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("GIRO_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
if not DATA_DIR.is_absolute():
    DATA_DIR = PROJECT_ROOT / DATA_DIR
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

for _d in (DATA_DIR, INTERIM_DIR, PROCESSED_DIR, MLRUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

RAW_FILES = {
    "clientes": "clientes.csv",
    "produtos": "produtos.csv",
    "pedidos": "pedidos.csv",
    "itens_pedido": "itens_pedido.csv",
    "precos_produto": "precos_produto.csv",
    "vendedores": "vendedores.csv",
}

