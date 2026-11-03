"""Data drift detection with Evidently (report + summary)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import PROCESSED_DIR


def build_reference_current(reference: pd.DataFrame, current: pd.DataFrame,
                            numeric_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return reference[numeric_cols].copy(), current[numeric_cols].copy()


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                     numeric_cols: list[str], out_html: Path | None = None) -> dict:
    ref, cur = build_reference_current(reference, current, numeric_cols)
    mapping = ColumnMapping(numerical_features=numeric_cols)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur, column_mapping=mapping)
    if out_html:
        report.save_html(str(out_html))
    else:
        out_html = PROCESSED_DIR / "drift_report.html"
        report.save_html(str(out_html))
    result = report.as_dict()
    drift = result["metrics"][0]["result"]
    return {
        "dataset_drift": bool(drift.get("dataset_drift", False)),
        "share_drifted": float(drift.get("share_of_drifted_columns", 0.0)),
        "report_path": str(out_html),
    }
