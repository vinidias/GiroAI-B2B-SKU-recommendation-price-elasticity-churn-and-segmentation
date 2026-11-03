import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("D:/Dev/giro-ai/data/raw")
files = sorted(RAW.glob("*.csv"))
print(f"{'arquivo':32s} {'linhas':>9s} {'cols':>5s}")
print("-" * 50)
for f in files:
    df = pd.read_csv(f, encoding="utf-8-sig", dtype=str, keep_default_na=False, nrows=1)
    with open(f, encoding="utf-8-sig") as fh:
        n = sum(1 for _ in fh) - 1
    print(f"{f.name:32s} {n:>9,} {len(df.columns):>5}")